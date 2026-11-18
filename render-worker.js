/* Iris render worker.
   Each worker builds the polar "unwrapped" iris from the photo, then renders a horizontal band
   of the final image. Several workers run in parallel, one band each. */
"use strict";

let seedInt = 0;

function hash3(x, y, z) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 1274126177) + Math.imul(seedInt, 1013904223)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16;
  return (h & 0xffff) / 65535;
}
function noise3(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  const a = hash3(xi, yi, zi), b = hash3(xi + 1, yi, zi), c = hash3(xi, yi + 1, zi), d = hash3(xi + 1, yi + 1, zi);
  const e = hash3(xi, yi, zi + 1), f = hash3(xi + 1, yi, zi + 1), g = hash3(xi, yi + 1, zi + 1), h = hash3(xi + 1, yi + 1, zi + 1);
  const x1 = a + (b - a) * u, x2 = c + (d - c) * u, x3 = e + (f - e) * u, x4 = g + (h - g) * u;
  const y1 = x1 + (x2 - x1) * v, y2 = x3 + (x4 - x3) * v;
  return y1 + (y2 - y1) * w;
}
const sstep = (a, b, x) => { let t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
const lumOf = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

const NT = 720, NR = 128, N = NT * NR;

function buildFields(img, cx, cy, R, pr) {
  const { w, h, data } = img;
  const o = [0, 0, 0];
  const sample = (x, y) => {
    x = clamp(x, 0, w - 1.001); y = clamp(y, 0, h - 1.001);
    const xi = x | 0, yi = y | 0, fx = x - xi, fy = y - yi;
    const i00 = (yi * w + xi) * 4, i10 = i00 + 4, i01 = i00 + w * 4, i11 = i01 + 4;
    for (let c = 0; c < 3; c++) {
      const a = data[i00 + c] + (data[i10 + c] - data[i00 + c]) * fx, b = data[i01 + c] + (data[i11 + c] - data[i01 + c]) * fx;
      o[c] = (a + (b - a) * fy) / 255;
    }
  };
  const r = R * pr;
  const pr_ = new Float32Array(N), pg = new Float32Array(N), pb = new Float32Array(N), pl = new Float32Array(N);
  // 1) unwrap the iris into a rectangle (angle x radius), staying just inside both edges
  for (let i = 0; i < NT; i++) {
    const a = i / NT * 2 * Math.PI - Math.PI, ca = Math.cos(a), sa = Math.sin(a);
    for (let j = 0; j < NR; j++) {
      const rho = r * 1.04 + (R * 0.93 - r * 1.04) * (j + 0.5) / NR;
      sample(cx + ca * rho, cy + sa * rho);
      const k = j * NT + i; pr_[k] = o[0]; pg[k] = o[1]; pb[k] = o[2]; pl[k] = lumOf(o[0], o[1], o[2]);
    }
  }
  // 2) per ring, replace outliers (eyelids, lashes, glare) by the ring's median colour
  let gl = 0;
  const tmpSort = new Float32Array(NT), dist = new Float32Array(NT);
  const med = (arr, b) => { tmpSort.set(arr.subarray(b, b + NT)); tmpSort.sort(); return tmpSort[NT >> 1]; };
  for (let j = 0; j < NR; j++) {
    const b = j * NT;
    const mr = med(pr_, b), mg = med(pg, b), mb = med(pb, b);
    for (let i = 0; i < NT; i++) { const k = b + i; dist[i] = Math.abs(pr_[k] - mr) + Math.abs(pg[k] - mg) + Math.abs(pb[k] - mb); }
    tmpSort.set(dist); tmpSort.sort();
    const thr = Math.max(0.05, 2.8 * tmpSort[NT >> 1]);
    for (let i = 0; i < NT; i++) { const k = b + i; if (dist[i] > thr) { pr_[k] = mr; pg[k] = mg; pb[k] = mb; pl[k] = lumOf(mr, mg, mb); } }
    gl += lumOf(mr, mg, mb);
  }
  gl /= NR;
  // 3) low-frequency colour field (the person's real colour layout) + real fine detail
  const blur = (arr, rt, rr) => {
    const tmp = new Float32Array(N);
    for (let j = 0; j < NR; j++) {
      const b = j * NT; let s = 0;
      for (let k = -rt; k <= rt; k++) s += arr[b + ((k + NT) % NT)];
      for (let i = 0; i < NT; i++) { tmp[b + i] = s / (2 * rt + 1); s += arr[b + ((i + rt + 1) % NT)] - arr[b + ((i - rt + NT) % NT)]; }
    }
    for (let i = 0; i < NT; i++) for (let j = 0; j < NR; j++) {
      let s = 0, n = 0;
      for (let k = Math.max(0, j - rr); k <= Math.min(NR - 1, j + rr); k++) { s += tmp[k * NT + i]; n++; }
      arr[j * NT + i] = s / n;
    }
  };
  const br = pr_.slice(), bg = pg.slice(), bb = pb.slice(), bl = pl.slice();
  for (let p = 0; p < 2; p++) { blur(br, 10, 4); blur(bg, 10, 4); blur(bb, 10, 4); blur(bl, 10, 4); }
  const det = new Float32Array(N);
  for (let k = 0; k < N; k++) det[k] = clamp((pl[k] - bl[k]) / (bl[k] + 0.05), -0.6, 0.6);
  const expo = clamp(0.36 / Math.max(0.02, gl), 0.8, 3.2);
  return { br, bg, bb, det, expo };
}

function renderRows(F, S, pr, start, step) {
  const { br, bg, bb, det, expo } = F;
  const field = (t, u, arr) => {
    let fi = t * NT - 0.5; const fj = clamp(u * NR - 0.5, 0, NR - 1.001);
    if (fi < 0) fi += NT;
    const i0 = fi | 0, i1 = (i0 + 1) % NT, ft = fi - i0, j0 = fj | 0, j1 = Math.min(NR - 1, j0 + 1), fr = fj - j0;
    const a = arr[j0 * NT + i0] + (arr[j0 * NT + i1] - arr[j0 * NT + i0]) * ft;
    const b = arr[j1 * NT + i0] + (arr[j1 * NT + i1] - arr[j1 * NT + i0]) * ft;
    return a + (b - a) * fr;
  };
  const FIB = 1.0, REAL = 1.2, AMBER = 0.35, SAT = 1.35, F1 = 55, F2 = 130;
  const C = S / 2, Ro = S * 0.41, ro = Ro * pr;
  const clX = C - Ro * 0.33, clY = C - Ro * 0.40, clR = Ro * 0.10;
  const rows = Math.ceil((S - start) / step);
  const out = new Uint8ClampedArray(rows * S * 4);
  let rnd = ((start + 1) * 2654435761) >>> 0 || 1;
  for (let y = start, row = 0; y < S; y += step, row++) {
    for (let x = 0; x < S; x++) {
      const dx = x + 0.5 - C, dy = y + 0.5 - C, d = Math.sqrt(dx * dx + dy * dy);
      let R_ = 0, G_ = 0, B_ = 0;
      if (d < Ro * 1.06) {
        const ca = d > 0 ? dx / d : 1, sa = d > 0 ? dy / d : 0;
        const uc = clamp((d - ro) / (Ro - ro), 0, 1);
        const t = (Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI);
        let r0 = field(t, uc, br) * expo, g0 = field(t, uc, bg) * expo, b0 = field(t, uc, bb) * expo;
        // wavy radial fibres
        const wob = (noise3(ca * 3, sa * 3, uc * 2.5 + 11) - 0.5) * 0.09;
        const c2 = ca - sa * wob, s2 = sa + ca * wob;
        const n1 = noise3(c2 * F1, s2 * F1, uc * 2.6), n1b = noise3(c2 * F1 * 2.03, s2 * F1 * 2.03, uc * 5.1 + 3);
        const n2 = noise3(c2 * F2, s2 * F2, uc * 6 + 7);
        let rid1 = 1 - Math.abs(2 * (n1 * 0.7 + n1b * 0.3) - 1); rid1 = rid1 * rid1 * rid1;
        let rid2 = 1 - Math.abs(2 * n2 - 1); rid2 = rid2 * rid2 * rid2 * rid2;
        // collarette ring
        const cu = 0.30 + (noise3(ca * 4, sa * 4, 21) - 0.5) * 0.14 + (noise3(ca * 14, sa * 14, 29) - 0.5) * 0.04;
        const ridge = Math.exp(-(((uc - cu) / 0.022) ** 2));
        const inner = sstep(cu + 0.03, cu - 0.03, uc);
        let L = 0.62 + FIB * (0.85 * rid1 + 0.4 * rid2 - 0.42);
        // crypts and contraction furrows
        const cr = noise3(ca * 16, sa * 16, uc * 5 + 13);
        const zone = sstep(cu - 0.06, cu + 0.02, uc) * sstep(0.8, 0.55, uc);
        const crypt = sstep(0.63, 0.76, cr) * zone;
        let fur = 0;
        for (const uk of [0.6, 0.73, 0.86]) {
          const w = (noise3(ca * 6, sa * 6, uk * 10 + 31) - 0.5) * 0.035;
          const arc = sstep(0.38, 0.62, noise3(ca * 3 + uk * 5, sa * 3, 57));
          fur += Math.exp(-(((uc - uk - w) / 0.008) ** 2)) * arc;
        }
        L *= 1 - 0.6 * crypt; L *= 1 - 0.3 * fur;
        L += 0.32 * ridge;
        L *= 1 + field(t, uc, det) * REAL * 0.9;
        const am = inner * AMBER;
        r0 *= L * (1 + 0.32 * am); g0 *= L * (1 + 0.02 * am); b0 *= L * (1 - 0.38 * am);
        const lk = 1 - 0.82 * sstep(0.76, 1.0, uc); r0 *= lk; g0 *= lk; b0 *= lk * 1.04;
        const ruff = sstep(0.05, 0.0, uc) * 0.85;
        r0 = r0 + (0.16 - r0) * ruff; g0 = g0 + (0.08 - g0) * ruff; b0 = b0 + (0.035 - b0) * ruff;
        const gy = lumOf(r0, g0, b0); r0 = gy + (r0 - gy) * SAT; g0 = gy + (g0 - gy) * SAT; b0 = gy + (b0 - gy) * SAT;
        const pin = sstep(ro - 1.2, ro + 1.2, d);
        const pupil = 0.008 + 0.01 * (1 - d / ro);
        const outE = sstep(Ro + S * 0.012, Ro - 1, d);
        R_ = (pupil + (r0 - pupil) * pin) * outE; G_ = (pupil + (g0 - pupil) * pin) * outE; B_ = (pupil + (b0 - pupil) * pin) * outE;
        // light reflection
        const ex = (x - clX) / (clR * 1.25), ey = (y - clY) / clR, e = Math.sqrt(ex * ex + ey * ey);
        let cl = sstep(1.0, 0.75, e) * 0.95 + sstep(2.2, 0.9, e) * 0.08;
        const e2x = (x - (C + Ro * 0.42)) / (clR * 0.35), e2y = (y - (C + Ro * 0.30)) / (clR * 0.35);
        cl += sstep(1, 0.5, Math.sqrt(e2x * e2x + e2y * e2y)) * 0.5;
        cl = Math.min(1, cl) * outE;
        R_ = 1 - (1 - R_) * (1 - cl); G_ = 1 - (1 - G_) * (1 - cl); B_ = 1 - (1 - B_) * (1 - cl);
      }
      rnd ^= rnd << 13; rnd ^= rnd >>> 17; rnd ^= rnd << 5;
      const gr = ((rnd >>> 0) / 4294967296 - 0.5) * 0.018;
      const k = (row * S + x) * 4;
      out[k] = (R_ + gr) * 255; out[k + 1] = (G_ + gr) * 255; out[k + 2] = (B_ + gr) * 255; out[k + 3] = 255;
    }
  }
  return out;
}

// Bake exposure into the colour fields so two eyes can be blended on equal footing.
function bake(F) {
  for (let k = 0; k < N; k++) { F.br[k] *= F.expo; F.bg[k] *= F.expo; F.bb[k] *= F.expo; }
  F.expo = 1; return F;
}

// Fusion: eye A on one half, eye B on the other, meeting along a soft, wavy seam (like sectoral heterochromia).
function fuse(A, B, swap) {
  const out = { br: new Float32Array(N), bg: new Float32Array(N), bb: new Float32Array(N), det: new Float32Array(N), expo: 1 };
  const side = swap ? 1 : -1;
  for (let i = 0; i < NT; i++) {
    const a = i / NT * 2 * Math.PI - Math.PI, ca = Math.cos(a), sa = Math.sin(a);
    for (let j = 0; j < NR; j++) {
      const u = (j + 0.5) / NR;
      const wob = (noise3(sa * 3 + 5, u * 4, 77) - 0.5) * 0.5 + (noise3(sa * 11, u * 9, 91) - 0.5) * 0.12;
      const w = sstep(-0.07, 0.07, side * ca + wob);
      const k = j * NT + i;
      out.br[k] = A.br[k] * w + B.br[k] * (1 - w);
      out.bg[k] = A.bg[k] * w + B.bg[k] * (1 - w);
      out.bb[k] = A.bb[k] * w + B.bb[k] * (1 - w);
      out.det[k] = A.det[k] * w + B.det[k] * (1 - w);
    }
  }
  return out;
}

self.onmessage = (e) => {
  // rows are interleaved across workers (start, start+step, ...) so every worker gets an equal share of iris
  const { id, A, B, swap, seed, S, start, step } = e.data;
  seedInt = seed | 0;
  let F = bake(buildFields(A.img, A.cx, A.cy, A.R, A.pr)), pr = A.pr;
  if (B) {
    F = fuse(F, bake(buildFields(B.img, B.cx, B.cy, B.R, B.pr)), swap);
    pr = (A.pr + B.pr) / 2;
  }
  const band = renderRows(F, S, pr, start, step);
  self.postMessage({ id, start, step, band }, [band.buffer]);
};
