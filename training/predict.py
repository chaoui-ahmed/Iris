"""Run the exported model on your own photos and save overlays, to check it before deploying.

    python predict.py photos/*.jpg --out preds/

Uses the same two-pass logic and circle fit as the website (index.html), so what you see here is what
the site will do.
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 128


def run(sess, crop):
    x = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return sess.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0]          # [4,128,128]


def fit(probs):
    """Iris circle from the iris+pupil region's left/right edges (lids hide top and bottom), pupil from its area.
    Returns dict in 128-px coords or None."""
    lab = probs.argmax(0)
    disk = lab >= 2
    n = int(disk.sum())
    if n < 40:
        return None
    pts = []
    for y in range(SIZE):
        xs = np.nonzero(disk[y])[0]
        if len(xs) >= 2:
            for x in (xs[0], xs[-1]):
                if 0 < x < SIZE - 1:
                    pts.append((x + 0.5, y + 0.5))
    if len(pts) < 12:
        return None
    P = np.array(pts, np.float64)

    def kasa(P):
        A = np.c_[2 * P, np.ones(len(P))]
        b = (P ** 2).sum(1)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        r = np.sqrt(max(1e-6, c[2] + c[0] ** 2 + c[1] ** 2))
        return c[0], c[1], r

    cx, cy, r = kasa(P)
    res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
    keep = res < max(1.5, 2.5 * np.median(res))
    if keep.sum() >= 12:
        cx, cy, r = kasa(P[keep])
        res = np.abs(np.hypot(P[keep, 0] - cx, P[keep, 1] - cy) - r)
    pup = lab == 3
    pn = int(pup.sum())
    if pn > 6:
        ys, xs = np.nonzero(pup)
        pcx, pcy, pr_ = xs.mean() + 0.5, ys.mean() + 0.5, np.sqrt(pn / np.pi)
        ratio = float(np.clip(pr_ / r, 0.15, 0.7))
        centred = np.hypot(pcx - cx, pcy - cy) < 0.35 * r
    else:
        ratio, centred = 0.36, False
    # confidence: confident pixels, round shape, a visible part of the circle, sclera next to it, centred pupil
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    circ = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
    cover = (disk & circ).sum() / max(1, circ.sum())
    spill = (disk & ~circ).sum() / max(1, n)
    sure = probs[2:].sum(0)[disk].mean()
    ring = ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= (1.3 * r) ** 2) & ~circ
    sclera = (lab[ring] == 1).mean() if ring.any() else 0
    roundness = float(np.mean(res) / r)
    conf = float(np.clip((cover - 0.3) / 0.4, 0, 1) * np.clip((0.12 - roundness) / 0.08, 0, 1)
                 * np.clip((sure - 0.6) / 0.3, 0, 1) * np.clip(sclera / 0.25, 0, 1) * (1 if centred else 0.6 if pn > 6 else 0.3)
                 * np.clip((0.25 - spill) / 0.15, 0, 1))
    return dict(cx=cx, cy=cy, r=r, pr=ratio, conf=conf, ok=conf > 0.35 and r > 3)


def detect(sess, img):
    """Pass 1 on the whole (letterboxed) image, pass 2 on a crop around the iris."""
    H, W = img.shape[:2]
    side = max(H, W)
    box = np.zeros((side, side, 3), img.dtype)
    oy, ox = (side - H) // 2, (side - W) // 2
    box[oy:oy + H, ox:ox + W] = img
    f = fit(run(sess, box))
    if not f:
        return None
    s = side / SIZE
    cx, cy, r = f["cx"] * s - ox, f["cy"] * s - oy, f["r"] * s
    c2 = r * 2.6
    M = np.float32([[SIZE / c2, 0, SIZE / 2 - cx * SIZE / c2], [0, SIZE / c2, SIZE / 2 - cy * SIZE / c2]])
    crop = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderValue=0)
    g = fit(run(sess, crop))
    if not g:
        return dict(cx=cx, cy=cy, r=r, pr=f["pr"], conf=f["conf"] * 0.5, ok=False)
    k = c2 / SIZE
    return dict(cx=cx + (g["cx"] - SIZE / 2) * k, cy=cy + (g["cy"] - SIZE / 2) * k, r=g["r"] * k, pr=g["pr"],
                conf=g["conf"], ok=g["ok"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "iris-seg.onnx"))
    ap.add_argument("--out", default="preds")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            continue
        rgb = img[..., ::-1].copy()
        d = detect(sess, rgb)
        vis = img.copy()
        if d:
            col = (80, 220, 120) if d["ok"] else (60, 60, 255)
            t = max(2, int(max(img.shape[:2]) / 400))
            cv2.circle(vis, (int(d["cx"]), int(d["cy"])), int(d["r"]), col, t, cv2.LINE_AA)
            cv2.circle(vis, (int(d["cx"]), int(d["cy"])), int(d["r"] * d["pr"]), col, t, cv2.LINE_AA)
            msg = f"{'ok' if d['ok'] else 'unsure'} conf={d['conf']:.2f}"
        else:
            msg = "no iris"
        cv2.putText(vis, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, img.shape[1] / 1000), (255, 255, 255), 2)
        out = os.path.join(a.out, os.path.basename(p))
        cv2.imwrite(out, vis)
        print(f"{p}: {msg}")
