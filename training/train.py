"""Train the iris segmenter and export it for the website.

    python train.py                                   # synthetic data only (works out of the box)
    python train.py --real data/real --epochs 30      # add a real dataset (see README)

Writes checkpoints/best.pt and ../models/iris-seg.onnx (what index.html loads).
Runs on CPU (slow but fine for synthetic), CUDA (Colab/Kaggle) or Apple MPS.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from model import IrisNet
from synth_eyes import generate

SIZE = 128


def _gen_chunk(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    ims, ms = zip(*(generate(rng, SIZE) for _ in range(n)))
    return np.stack(ims), np.stack(ms)


def synth_cache(n, seed, path, workers):
    """Pre-generate synthetic samples once (generation is slower than training a step on CPU)."""
    if os.path.exists(path):
        d = np.load(path)
        if len(d["x"]) >= n:
            return d["x"][:n], d["m"][:n]
    chunks = [(seed + i, min(500, n - i * 500)) for i in range((n + 499) // 500)]
    with Pool(workers) as p:
        parts = p.map(_gen_chunk, chunks)
    x = np.concatenate([a for a, _ in parts])
    m = np.concatenate([b for _, b in parts])
    np.savez(path, x=x, m=m)
    return x, m


def augment(img, mask, rng):
    """Light augmentation shared by both sources: letterboxing (as the site does for whole photos),
    colour jitter, blur, flip."""
    if rng.random() < 0.2:                                   # letterbox into a random sub-rectangle
        s = rng.uniform(0.5, 0.95)
        h = w = SIZE
        if rng.random() < 0.5:
            nh, nw = int(h * s), w
        else:
            nh, nw = h, int(w * s)
        y0, x0 = rng.integers(0, h - nh + 1), rng.integers(0, w - nw + 1)
        bi, bm = np.zeros_like(img), np.zeros_like(mask)
        bi[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        bm[y0:y0 + nh, x0:x0 + nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        img, mask = bi, bm
    img = img.astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-15, 15)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    return np.clip(img, 0, 255), np.ascontiguousarray(mask)


class ArrayDS(Dataset):
    def __init__(self, x, m, train=True):
        self.x, self.m, self.train = x, m, train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img, mask = self.x[i], self.m[i]
        if self.train:
            img, mask = augment(img, mask, np.random.default_rng())
        img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1) / 255
        return img, torch.from_numpy(mask.astype(np.int64))


class RealDS(Dataset):
    """data/real/images/<name>.(jpg|png) + data/real/masks/<name>.png with values 0..3
    (0 background, 1 sclera, 2 iris, 3 pupil). Use 255 for "unknown" pixels: they're ignored in the loss.
    Random crops around the eye at different scales mimic both detection passes of the site."""

    def __init__(self, root, train=True, repeat=4):
        self.items = []
        for p in sorted(glob.glob(os.path.join(root, "images", "*"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            mp = os.path.join(root, "masks", stem + ".png")
            if os.path.exists(mp):
                self.items.append((p, mp))
        self.train, self.repeat = train, repeat if train else 1

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        p, mp = self.items[i % len(self.items)]
        img = cv2.imread(p)[..., ::-1]
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        rng = np.random.default_rng()
        H, W = mask.shape
        ys, xs = np.nonzero((mask == 2) | (mask == 3))
        if len(xs) and self.train:
            cx, cy = xs.mean() + rng.uniform(-0.3, 0.3) * W * 0.1, ys.mean() + rng.uniform(-0.3, 0.3) * H * 0.1
            r = max(xs.ptp(), ys.ptp()) / 2 + 1
            side = r * rng.uniform(2.2, 8.0)
        else:
            cx, cy, side = W / 2, H / 2, max(W, H)
        side = max(8, side)
        M = np.float32([[SIZE / side, 0, SIZE / 2 - cx * SIZE / side], [0, SIZE / side, SIZE / 2 - cy * SIZE / side]])
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpAffine(mask, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)
        if self.train:
            img, mask = augment(img, mask, rng)
        img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1) / 255
        return img, torch.from_numpy(mask.astype(np.int64))


def dice_loss(logits, target, classes=(2, 3)):
    probs = logits.softmax(1)
    valid = (target != 255).float()
    loss = 0
    for c in classes:
        p = probs[:, c] * valid
        t = (target == c).float()
        inter = (p * t).sum((1, 2))
        loss = loss + 1 - ((2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()
    return loss / len(classes)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    inter = np.zeros(4)
    union = np.zeros(4)
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)
        disk_p, disk_t = pred >= 2, y >= 2                   # iris + pupil = the iris disk
        for k, (a, b) in enumerate([(disk_p, disk_t), (pred == 3, y == 3), (pred == 1, y == 1)]):
            v = y != 255
            inter[k] += ((a & b) & v).sum().item()
            union[k] += ((a | b) & v).sum().item()
    model.train()
    iou = inter[:3] / np.maximum(union[:3], 1)
    return {"iris_disk": iou[0], "pupil": iou[1], "sclera": iou[2]}


class Export(nn.Module):
    """What the browser runs: uint8-range RGB in, class probabilities out."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):                                    # x: [1,3,128,128] in 0..1
        return self.net(x).softmax(1)


def export_onnx(model, path):
    model = model.cpu().eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(Export(model), torch.zeros(1, 3, SIZE, SIZE), path, input_names=["image"],
                      output_names=["probs"], opset_version=17, dynamo=False)
    print("exported", path, f"{os.path.getsize(path) / 1024:.0f} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=8000, help="number of synthetic samples (0 to disable)")
    ap.add_argument("--real", default=None, help="folder with images/ and masks/ (train); optional")
    ap.add_argument("--real-val", default=None, help="held-out real folder for validation; optional")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", default=None, help="start from a checkpoint (fine-tuning)")
    ap.add_argument("--seed", type=int, default=1000, help="seed for the synthetic set")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "iris-seg.onnx"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs("checkpoints", exist_ok=True)
    t0 = time.time()

    train_sets, val_sets = [], []
    if a.synthetic:
        x, m = synth_cache(a.synthetic, a.seed, f"checkpoints/synth_{a.synthetic}_{a.seed}.npz", a.workers)
        vx, vm = synth_cache(600, 999_000, "checkpoints/synth_val.npz", a.workers)
        train_sets.append(ArrayDS(x, m))
        val_sets.append(ArrayDS(vx, vm, train=False))
        print(f"synthetic ready: {len(x)} train / {len(vx)} val ({time.time() - t0:.0f}s)")
    if a.real:
        train_sets.append(RealDS(a.real))
        print("real train images:", len(train_sets[-1].items))
    if a.real_val:
        val_sets.append(RealDS(a.real_val, train=False))
    train = DataLoader(ConcatDataset(train_sets), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                       drop_last=True, persistent_workers=a.workers > 0)
    val = DataLoader(ConcatDataset(val_sets), batch_size=64, num_workers=0)

    model = IrisNet().to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train))
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0], device=dev)
    best = -1
    for ep in range(a.epochs):
        tl, n = 0.0, 0
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, ignore_index=255) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
            n += 1
        res = evaluate(model, val, dev)
        score = res["iris_disk"] + 0.5 * res["pupil"]
        print(f"epoch {ep + 1}/{a.epochs}  loss {tl / n:.3f}  IoU iris {res['iris_disk']:.3f} "
              f"pupil {res['pupil']:.3f} sclera {res['sclera']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), "checkpoints/best.pt")
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=dev))
    export_onnx(model, os.path.abspath(a.out))


if __name__ == "__main__":
    main()
