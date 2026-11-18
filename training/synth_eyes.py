"""Procedural eye generator: images + pixel masks for training the iris segmenter.

Labels: 0 background, 1 sclera (white of the eye), 2 iris, 3 pupil.
Eyelids hide parts of the iris; eyelashes and reflections are drawn over it but keep the label underneath,
so the model learns to "see through" them (what we need to fit circles).

Real photos are always better; this exists so a usable model can be trained before any dataset arrives,
and to add variety (skin tones, eye colours, lighting, blur) on top of real data later.

    python synth_eyes.py --out samples --n 16      # writes a preview grid
"""
import argparse
import math

import cv2
import numpy as np

SKIN = [(236, 200, 175), (224, 172, 140), (198, 140, 105), (160, 105, 75), (120, 78, 55), (88, 58, 42), (60, 40, 30)]
IRIS = {  # (inner/pupillary colour, outer colour, streak colour) in RGB
    "brown": [(110, 62, 25), (75, 42, 20), (150, 95, 45)],
    "dark": [(55, 32, 18), (40, 24, 14), (90, 55, 30)],
    "hazel": [(150, 100, 40), (95, 105, 60), (190, 160, 90)],
    "green": [(140, 120, 50), (70, 110, 75), (160, 190, 130)],
    "blue": [(120, 130, 120), (60, 100, 150), (170, 200, 230)],
    "grey": [(120, 120, 115), (95, 110, 120), (190, 200, 205)],
    "amber": [(170, 110, 40), (140, 90, 35), (220, 170, 90)],
}


def _noise2d(rng, h, w, scale):
    """Smooth value noise in [0,1]."""
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def _periodic1d(rng, n, length):
    """Random smooth periodic function of angle sampled at `n` points, built from `length` knots."""
    k = rng.standard_normal(length).astype(np.float32)
    k = np.concatenate([k, k[:4]])
    x = np.linspace(0, length, n, endpoint=False)
    i = np.floor(x).astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return k[i] * (1 - f) + k[i + 1] * f


def _skin(rng, H, W):
    base = np.array(SKIN[rng.integers(len(SKIN))], np.float32) * rng.uniform(0.85, 1.12, 3)
    img = np.ones((H, W, 3), np.float32) * base
    shade = 0.75 + 0.5 * _noise2d(rng, H, W, rng.uniform(60, 200))
    pores = 0.94 + 0.12 * _noise2d(rng, H, W, rng.uniform(2, 5))
    return img * (shade * pores)[..., None]


def _iris_texture(rng, size, r_out, r_pup):
    """Square RGB iris texture centred in a size x size patch, plus the polar coords used."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    dx, dy = xx - c, yy - c
    rho = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    u = ((rho - r_pup) / max(1.0, r_out - r_pup)).clip(0, 1)
    kind = list(IRIS)[rng.integers(len(IRIS))]
    inner, outer, streak = (np.array(v, np.float32) * rng.uniform(0.75, 1.25) for v in IRIS[kind])
    ti = ((th + math.pi) / (2 * math.pi) * 2048).astype(int) % 2048
    fib = np.zeros_like(u)
    for knots, amp in ((rng.integers(60, 140), 1.0), (rng.integers(150, 400), 0.6), (rng.integers(20, 50), 0.5)):
        f = _periodic1d(rng, 2048, int(knots))
        fib += amp * f[ti] * (0.6 + 0.4 * np.cos(u * rng.uniform(2, 6) + rng.uniform(0, 6)))
    fib = (fib - fib.min()) / (np.ptp(fib) + 1e-6)
    collarette = rng.uniform(0.2, 0.42) + 0.06 * _periodic1d(rng, 2048, int(rng.integers(5, 12)))[ti]
    inner_w = (1 - np.clip((u - collarette) / 0.05, 0, 1))[..., None]
    col = inner * inner_w + outer * (1 - inner_w)
    col = col * (0.6 + 0.8 * fib[..., None]) + streak * (0.25 * fib[..., None] ** 3)
    ridge = np.exp(-((u - collarette) / 0.03) ** 2)[..., None]
    col = col * (1 + 0.25 * ridge)
    crypt = (_noise2d(rng, size, size, max(2, r_out / rng.uniform(5, 12))) > rng.uniform(0.7, 0.85)).astype(np.float32)
    col *= (1 - 0.4 * crypt * (u > collarette) * (u < 0.8))[..., None]
    limb = np.clip((u - rng.uniform(0.7, 0.88)) / 0.15, 0, 1) ** 1.5
    col *= (1 - rng.uniform(0.4, 0.85) * limb)[..., None]
    ruff = (u < 0.06).astype(np.float32)[..., None]
    col = col * (1 - 0.6 * ruff) + np.array([60, 30, 15], np.float32) * 0.6 * ruff
    return col, rho


def generate(rng, out_size=128, work=320):
    """Return (image uint8 HxWx3 RGB, mask uint8 HxW)."""
    H = W = work
    img = _skin(rng, H, W)
    mask = np.zeros((H, W), np.uint8)

    if rng.random() < 0.22:
        return _negative(rng, img, out_size)

    r_i = rng.uniform(0.06, 0.42) * work                      # iris radius
    a = r_i * rng.uniform(2.0, 2.8)                           # half eye width
    cx = rng.uniform(0.15, 0.85) * W
    cy = rng.uniform(0.2, 0.8) * H
    openness = rng.uniform(0.55, 1.15)
    hu = a * rng.uniform(0.38, 0.6) * openness
    hl = a * rng.uniform(0.22, 0.4) * openness
    tilt = rng.uniform(-0.12, 0.12)
    t = np.linspace(-1, 1, 64)
    up = np.stack([cx + a * t, cy - hu * (1 - t ** 2) ** 0.8 + tilt * a * t], 1)
    lo = np.stack([cx + a * t[::-1], cy + hl * (1 - t[::-1] ** 2) ** 0.9 + tilt * a * t[::-1]], 1)
    poly = np.concatenate([up, lo]).astype(np.float32)
    ang = rng.uniform(-18, 18)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    poly = (np.c_[poly, np.ones(len(poly))] @ M.T).astype(np.int32)
    aperture = np.zeros((H, W), np.uint8)
    cv2.fillPoly(aperture, [poly], 1)

    # sclera
    scl = np.array([238, 228, 222], np.float32) * rng.uniform(0.75, 1.05) + rng.uniform(-10, 10, 3)
    dist = cv2.distanceTransform(aperture, cv2.DIST_L2, 5)
    depth = np.clip(dist / (hu * 0.6 + 1), 0, 1)
    sclera_img = scl * (0.55 + 0.45 * depth)[..., None]
    veins = np.zeros((H, W), np.float32)
    for _ in range(rng.integers(0, 14)):
        p = np.array([cx + rng.uniform(-a, a), cy + rng.uniform(-hl, hl)])
        pts = [p.copy()]
        d = rng.uniform(0, 2 * math.pi)
        for _ in range(8):
            d += rng.uniform(-0.6, 0.6)
            p = p + np.array([math.cos(d), math.sin(d)]) * a * 0.06
            pts.append(p.copy())
        cv2.polylines(veins, [np.array(pts, np.int32)], False, float(rng.uniform(0.2, 0.7)), 1, cv2.LINE_AA)
    sclera_img = sclera_img * (1 - veins[..., None]) + np.array([170, 60, 60], np.float32) * veins[..., None]

    # iris position inside the aperture (gaze)
    ix = cx + rng.uniform(-0.35, 0.35) * a
    iy = cy + rng.uniform(-0.25, 0.2) * hu
    ix, iy = (np.array([ix, iy, 1.0]) @ M.T)
    pr = rng.uniform(0.2, 0.6)
    size = int(2 * r_i) + 3
    tex, rho = _iris_texture(rng, size, r_i, r_i * pr)
    squash = rng.uniform(0.88, 1.0)                           # slight off-axis view
    tex = cv2.resize(tex, (size, max(3, int(size * squash))))
    rho = cv2.resize(rho, (size, max(3, int(size * squash))))
    th, tw = tex.shape[:2]
    x0, y0 = int(round(ix - tw / 2)), int(round(iy - th / 2))
    iris_layer = np.zeros((H, W, 3), np.float32)
    iris_lab = np.zeros((H, W), np.uint8)
    pupil_col = np.array([8, 6, 6], np.float32) * rng.uniform(0.5, 2)
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x0 + tw), min(H, y0 + th)
    if xs1 > xs0 and ys1 > ys0:
        sub_t = tex[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        sub_r = rho[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        inside = sub_r <= r_i
        pupil = sub_r <= r_i * pr
        t_ = np.where(pupil[..., None], pupil_col, sub_t)
        iris_layer[ys0:ys1, xs0:xs1][inside] = t_[inside]
        lab = np.where(pupil, 3, np.where(inside, 2, 0)).astype(np.uint8)
        iris_lab[ys0:ys1, xs0:xs1] = lab

    eye_img = np.where((iris_lab > 0)[..., None], iris_layer, sclera_img)
    # soft limbus: blur the iris edge a touch into the sclera
    eye_img = cv2.GaussianBlur(eye_img, (0, 0), max(0.6, r_i / 60))
    # shadow from the upper lid
    lid_shadow = np.clip(1 - dist / (hu * rng.uniform(0.3, 0.7) + 1), 0, 1)
    top = np.zeros((H, W), np.float32)
    cv2.fillPoly(top, [poly[:64]], 1)
    eye_img *= (1 - rng.uniform(0.2, 0.6) * lid_shadow * (iris_lab > 0 | (aperture > 0)))[..., None]

    ap3 = cv2.GaussianBlur(aperture.astype(np.float32), (0, 0), 1.0)[..., None]
    img = img * (1 - ap3) + eye_img * ap3
    mask[(aperture > 0)] = 1
    mask[(aperture > 0) & (iris_lab == 2)] = 2
    mask[(aperture > 0) & (iris_lab == 3)] = 3

    # eyelashes (drawn over everything, labels kept)
    lash_col = rng.uniform(10, 45)
    n_l = int(rng.integers(0, 90))
    for k in range(n_l):
        j = int(rng.integers(4, 60))
        p = poly[j].astype(np.float32)
        q = poly[j + 1].astype(np.float32)
        nrm = np.array([q[1] - p[1], -(q[0] - p[0])], np.float32)
        nrm /= np.linalg.norm(nrm) + 1e-6
        if nrm[1] > 0:
            nrm = -nrm
        L = a * rng.uniform(0.15, 0.45)
        bend = rng.uniform(-0.4, 0.4)
        pts = [p + nrm * L * s + np.array([bend * L * s * s, L * 0.25 * s * s]) for s in np.linspace(-0.15, 1, 6)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (lash_col,) * 3, max(1, int(r_i / 40)), cv2.LINE_AA)

    # reflections: flash dot, ring light, window
    for _ in range(int(rng.integers(0, 4))):
        layer = np.zeros((H, W), np.float32)
        px, py = ix + rng.uniform(-0.8, 0.8) * r_i, iy + rng.uniform(-0.8, 0.8) * r_i
        kind = rng.random()
        if kind < 0.5:
            cv2.ellipse(layer, (int(px), int(py)), (max(1, int(r_i * rng.uniform(0.04, 0.22))),) * 2, 0, 0, 360, 1, -1)
        elif kind < 0.75:
            rr = int(r_i * rng.uniform(0.3, 0.8))
            cv2.circle(layer, (int(ix), int(iy)), rr, 1, max(1, int(r_i * 0.06)))
        else:
            w_, h_ = r_i * rng.uniform(0.1, 0.4), r_i * rng.uniform(0.1, 0.4)
            cv2.rectangle(layer, (int(px), int(py)), (int(px + w_), int(py + h_)), 1, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), max(0.5, r_i * rng.uniform(0.005, 0.03))) * (aperture > 0)
        img = img * (1 - layer[..., None]) + 250 * layer[..., None]

    img, mask = _finish(rng, img, mask, out_size)
    return img, mask


def _negative(rng, img, out_size):
    """Frames with no eye: skin, clutter, or dark circles on light backgrounds (to defeat 'any dark disk')."""
    H, W = img.shape[:2]
    kind = rng.random()
    if kind < 0.45:
        return _finish(rng, _decoy(rng, H, W), np.zeros((H, W), np.uint8), out_size)
    if kind < 0.65:
        img = np.ones_like(img) * rng.uniform(30, 230, 3) * (0.6 + 0.8 * _noise2d(rng, H, W, rng.uniform(20, 120)))[..., None]
    for _ in range(int(rng.integers(0, 25))):
        col = tuple(float(c) for c in rng.uniform(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.uniform(4, W * 0.3)), col, -1 if rng.random() < 0.7 else int(rng.integers(1, 6)), cv2.LINE_AA)
        else:
            x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
            cv2.rectangle(img, (x, y), (x + int(rng.uniform(5, W * 0.4)), y + int(rng.uniform(5, H * 0.4))), col, -1)
    return _finish(rng, img, np.zeros((H, W), np.uint8), out_size)


def _decoy(rng, H, W):
    """Hard negatives: round things that aren't eyes. Dark disks and rings on light walls, lamps, coins,
    buttons, bottle tops: uniform or lightly textured, no almond-shaped opening, no lashes, mostly no pupil."""
    bg = rng.uniform(150, 250, 3) if rng.random() < 0.7 else rng.uniform(20, 250, 3)
    img = np.ones((H, W, 3), np.float32) * bg * (0.75 + 0.5 * _noise2d(rng, H, W, rng.uniform(40, 200)))[..., None]
    if rng.random() < 0.4:                                    # some room structure: door, shelf, frame
        for _ in range(int(rng.integers(1, 5))):
            x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
            cv2.rectangle(img, (x, y), (x + int(rng.uniform(20, W * 0.5)), y + int(rng.uniform(20, H * 0.6))),
                          tuple(float(c) for c in rng.uniform(10, 200, 3)), -1)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy, r = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H, rng.uniform(0.05, 0.4) * W
        col = rng.uniform(10, 140, 3)
        kind = rng.random()
        if kind < 0.6:                                        # filled disk, maybe shaded or textured
            layer = np.zeros((H, W), np.float32)
            cv2.circle(layer, (int(cx), int(cy)), int(r), 1, -1, cv2.LINE_AA)
            tex = 0.8 + 0.4 * _noise2d(rng, H, W, rng.uniform(2, 30)) if rng.random() < 0.5 else 1.0
            img = img * (1 - layer[..., None]) + (col * tex if np.ndim(tex) == 0 else col * tex[..., None]) * layer[..., None]
            if rng.random() < 0.2:                            # concentric target, the hardest case
                cv2.circle(img, (int(cx), int(cy)), int(r * rng.uniform(0.2, 0.5)), tuple(float(c) for c in rng.uniform(0, 60, 3)), -1, cv2.LINE_AA)
        else:                                                 # ring
            cv2.circle(img, (int(cx), int(cy)), int(r), tuple(float(c) for c in col), int(rng.integers(2, max(3, int(r * 0.3)))), cv2.LINE_AA)
    return img


def _finish(rng, img, mask, out_size):
    H, W = img.shape[:2]
    # lighting, colour cast, exposure
    g = np.linspace(rng.uniform(0.6, 1.2), rng.uniform(0.6, 1.2), W, dtype=np.float32)
    img = img * g[None, :, None] * rng.uniform(0.85, 1.15, 3) * rng.uniform(0.55, 1.35)
    if rng.random() < 0.5:
        yy, xx = np.mgrid[0:H, 0:W]
        v = 1 - rng.uniform(0, 0.5) * (((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W / 2))
        img *= v[..., None]
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rng.uniform(0.8, 1.25)
    # blur (focus, motion), noise, jpeg
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 3.5))
    if rng.random() < 0.25:
        k = int(rng.integers(3, 15))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1), (k, k))
        img = cv2.filter2D(img, -1, ker / (ker.sum() + 1e-6))
    img = img + rng.standard_normal(img.shape).astype(np.float32) * rng.uniform(0, 12)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.imdecode(cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 95))])[1], 1)[..., ::-1]
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    img = cv2.resize(np.ascontiguousarray(img), (out_size, out_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(np.ascontiguousarray(mask), (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return img, mask


PALETTE = np.array([[0, 0, 0], [90, 90, 90], [60, 200, 255], [255, 80, 80]], np.uint8)


def preview(n, out, seed=0, size=128):
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(n):
        im, m = generate(rng, size)
        over = (0.6 * im + 0.4 * PALETTE[m]).astype(np.uint8)
        tiles.append(np.concatenate([im, over], 0))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    grid = np.zeros((rows * 2 * size, cols * size, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        grid[r * 2 * size:(r + 1) * 2 * size, c * size:(c + 1) * size] = t
    cv2.imwrite(out, grid[..., ::-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synth_preview.png")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    preview(a.n, a.out, a.seed)
    print("wrote", a.out)
