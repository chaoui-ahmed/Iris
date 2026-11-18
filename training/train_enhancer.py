"""Detail enhancer (step 2, needs your own data): learns to turn a soft phone-quality iris into a sharp macro one.

Training trick: you only need sharp macro iris photos. Each one is degraded on the fly to look like a phone
shot (blur, noise, compression, lower resolution, flash glare) and the network learns to undo it.

    data/macro/*.jpg      sharp iris close-ups you have the rights to (own shots, licensed, free-licence)

    python train_enhancer.py --data data/macro --epochs 40
    -> ../models/iris-enhance.onnx  (input/output: [1,3,256,256] RGB in 0..1)

Not wired into the site yet: once you have a model you're happy with, it replaces the procedural fibres
in render-worker.js (run it on the unwrapped iris crop before the final render).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SIZE = 256


def degrade(img, rng):
    """Make a sharp macro crop look like a phone capture of the same iris."""
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    s = rng.uniform(0.15, 0.5)                                         # phones see far fewer pixels
    x = cv2.resize(x, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.8:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.8))
    x = x + rng.standard_normal(x.shape).astype(np.float32) * rng.uniform(1, 10)
    x = np.clip(x * rng.uniform(0.7, 1.2) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 85))])[1], 1)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.4:                                             # flash glare spot
        layer = np.zeros((h, w), np.float32)
        cv2.circle(layer, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(3, w * 0.06)), 1, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 2)[..., None]
        x = (x * (1 - layer) + 255 * layer).astype(np.uint8)
    return x


class MacroDS(Dataset):
    def __init__(self, root, repeat=20):
        self.paths = sorted(p for e in ("jpg", "jpeg", "png", "webp") for p in glob.glob(os.path.join(root, f"*.{e}")))
        assert self.paths, f"no images in {root}"
        self.repeat = repeat

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = cv2.imread(self.paths[i % len(self.paths)])
        h, w = img.shape[:2]
        side = int(min(h, w) * rng.uniform(0.35, 1.0))
        y, x = rng.integers(0, h - side + 1), rng.integers(0, w - side + 1)
        crop = cv2.resize(img[y:y + side, x:x + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        crop = np.ascontiguousarray(np.rot90(crop, int(rng.integers(4))))
        low = degrade(crop, rng)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[..., ::-1], dtype=np.float32) / 255).permute(2, 0, 1)
        return t(low), t(crop)


class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, 1, 1), nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class Enhancer(nn.Module):
    """Small residual net working at 1/2 resolution, predicting the detail to add on top of the input."""

    def __init__(self, c=48, n=8):
        super().__init__()
        self.inp = nn.Conv2d(3, c, 3, 2, 1)
        self.body = nn.Sequential(*[RB(c) for _ in range(n)])
        self.out = nn.Sequential(nn.Conv2d(c, 12, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, x):
        return (x + self.out(self.body(F.relu(self.inp(x))))).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/macro")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "iris-enhance.onnx"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dl = DataLoader(MacroDS(a.data), batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
    net = Enhancer().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    os.makedirs("checkpoints", exist_ok=True)
    for ep in range(a.epochs):
        tot = 0
        for lo, hi in dl:
            lo, hi = lo.to(dev), hi.to(dev)
            pred = net(lo)
            # L1 on pixels + L1 on image gradients (rewards sharp fibres, not just matching colours)
            gx = lambda t: t[..., :, 1:] - t[..., :, :-1]
            gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
            loss = F.l1_loss(pred, hi) + 0.5 * (F.l1_loss(gx(pred), gx(hi)) + F.l1_loss(gy(pred), gy(hi)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f}", flush=True)
        torch.save(net.state_dict(), "checkpoints/enhancer.pt")
    net = net.cpu().eval()
    torch.onnx.export(net, torch.zeros(1, 3, SIZE, SIZE), os.path.abspath(a.out), input_names=["image"],
                      output_names=["image_out"], opset_version=17, dynamo=False)
    print("exported", a.out)


if __name__ == "__main__":
    main()
