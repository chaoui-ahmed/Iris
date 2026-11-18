/* Iris render worker.
   Builds a polar "material" for each eye (colours from the user's photo + real iris structure from the
   texture library), then renders a share of the final image's rows. Several workers run in parallel. */
"use strict";

const NT = 1024, NR = 160, N = NT * NR;          // polar grid: angle x radius (pupil edge -> limbus)
let seedInt = 0;

const sstep = (a, b, x) => { let t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
const lumOf = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
function hash2(x, y) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seedInt, 1013904223)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16;
  return (h & 0xffff) / 65535;
}
function noise2(x, y) {                               // smooth value noise
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// separable box blur on the polar grid (wraps around in angle)
function blur(arr, rt, rr) {
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
  return arr;
}

// 1) the user's eye: unwrap, drop eyelids/lashes/glare per ring, keep the colour layout + real fine detail
function userFields(img, cx, cy, R, pr) {
  const { w, h, data } = img, o = [0, 0, 0];
  const sample = (x, y) => {
    x = clamp(x, 0, w - 1.001); y = clamp(y, 0, h - 1.001);
    const xi = x | 0, yi = y | 0, fx = x - xi, fy = y - yi;
    const i00 = (yi * w + xi) * 4, i10 = i00 + 4, i01 = i00 + w * 4, i11 = i01 + 4;
    for (let c = 0; c < 3; c++) {
      const a = data[i00 + c] + (data[i10 + c] - data[i00 + c]) * fx, b = data[i01 + c] + (data[i11 + c] - data[i01 + c]) * fx;
      o[c] = (a + (b - a) * fy) / 255;
    }
  };
  const r = R * pr, P = [new Float32Array(N), new Float32Array(N), new Float32Array(N)], L = new Float32Array(N);
  for (let i = 0; i < NT; i++) {
    const a = i / NT * 2 * Math.PI - Math.PI, ca = Math.cos(a), sa = Math.sin(a);
    for (let j = 0; j < NR; j++) {
      const rho = r * 1.06 + (R * 0.93 - r * 1.06) * (j + 0.5) / NR;
      sample(cx + ca * rho, cy + sa * rho);
      const k = j * NT + i; P[0][k] = o[0]; P[1][k] = o[1]; P[2][k] = o[2]; L[k] = lumOf(o[0], o[1], o[2]);
    }
  }
  const tmp = new Float32Array(NT), dist = new Float32Array(NT), rep = new Float32Array(N);
  const med = (arr, b) => { tmp.set(arr.subarray(b, b + NT)); tmp.sort(); return tmp[NT >> 1]; };
  let gl = 0;
  for (let j = 0; j < NR; j++) {
    const b = j * NT, m = [med(P[0], b), med(P[1], b), med(P[2], b)];
    for (let i = 0; i < NT; i++) { const k = b + i; dist[i] = Math.abs(P[0][k] - m[0]) + Math.abs(P[1][k] - m[1]) + Math.abs(P[2][k] - m[2]); }
    tmp.set(dist); tmp.sort();
    const thr = Math.max(0.05, 2.8 * tmp[NT >> 1]);
    for (let i = 0; i < NT; i++) { const k = b + i; if (dist[i] > thr) { P[0][k] = m[0]; P[1][k] = m[1]; P[2][k] = m[2]; L[k] = lumOf(m[0], m[1], m[2]); rep[k] = 1; } }
    gl += lumOf(m[0], m[1], m[2]);
  }
  gl /= NR;
  const base = P.map(a => blur(blur(a.slice(), 14, 6), 14, 6));
  const bl = blur(blur(L.slice(), 14, 6), 14, 6);
  const det = new Float32Array(N);
  for (let k = 0; k < N; k++) det[k] = clamp((L[k] - bl[k]) / (bl[k] + 0.05), -0.5, 0.5);
  // how much of each area the camera really saw (lids, lashes and glare were replaced above)
  const rel = blur(blur(rep, 30, 10), 30, 10);
  for (let k = 0; k < N; k++) { rel[k] = clamp(1 - 1.6 * rel[k], 0, 1); det[k] *= rel[k]; }
  return { base, det, gl, rel };
}

// 2) a library texture (real iris structure), recoloured to the user's colours
function material(U, lib) {
  const T = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
  for (let j = 0; j < NR; j++) for (let i = 0; i < NT; i++) {
    const k = j * NT + i, p = (j * lib.w + Math.floor(i * lib.w / NT)) * 4;
    T[0][k] = lib.data[p] / 255; T[1][k] = lib.data[p + 1] / 255; T[2][k] = lib.data[p + 2] / 255;
  }
  const tb = T.map(a => blur(blur(a.slice(), 22, 8), 22, 8));
  // per radial band colour ratio user/library, smoothed along the radius
  const ratio = [new Float32Array(NR), new Float32Array(NR), new Float32Array(NR)];
  for (let j = 0; j < NR; j++) {
    const j0 = Math.max(0, j - 12), j1 = Math.min(NR - 1, j + 12);
    for (let c = 0; c < 3; c++) {
      let su = 0, sl = 0;
      for (let jj = j0; jj <= j1; jj++) for (let i = 0; i < NT; i += 4) { su += U.base[c][jj * NT + i]; sl += tb[c][jj * NT + i]; }
      ratio[c][j] = clamp(su / Math.max(1e-4, sl), 0.15, 6);
    }
  }
  // exposure: keep light eyes light and dark eyes dark, but always well exposed
  const expo = clamp(0.4 / Math.max(0.03, U.gl), 0.9, 2.6) * (0.85 + 0.3 * clamp(U.gl / 0.4, 0, 1));
  const M = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
  for (let c = 0; c < 3; c++) {
    const dc = new Float32Array(N);
    for (let k = 0; k < N; k++) dc[k] = clamp(T[c][k] / Math.max(1e-3, tb[c][k]), 0.3, 2.4);
    const ds = blur(dc.slice(), 1, 1);
    for (let j = 0; j < NR; j++) for (let i = 0; i < NT; i++) {
      const k = j * NT + i;
      const d = dc[k] + 0.7 * (dc[k] - ds[k]);                         // sharpen the real fibres
      const libRecol = tb[c][k] * ratio[c][j];                          // library colour layout, user palette
      const wU = 0.55 * U.rel[k];                                      // trust the photo where it saw the iris
      const base = wU * U.base[c][k] + (1 - wU) * libRecol;
      M[c][k] = base * d * (1 + 0.35 * U.det[k]) * expo;
    }
  }
  return M;
}

// 3) fusion: two materials interlocking along the radial fibres, like two irises pushed into each other
function fuse(A, B, swap) {
  const out = [new Float32Array(N), new Float32Array(N), new Float32Array(N)], side = swap ? 1 : -1;
  for (let i = 0; i < NT; i++) {
    const a = i / NT * 2 * Math.PI - Math.PI, ca = Math.cos(a), sa = Math.sin(a), t = i / NT;
    for (let j = 0; j < NR; j++) {
      const u = (j + 0.5) / NR;
      // a swirling seam with interlocking tongues: the two irises look pushed into each other
      const warp = (noise2(t * 16, u * 2.5 + 3) - 0.5) * 1.3 + (noise2(t * 45, u * 5 + 9) - 0.5) * 0.55 + (noise2(t * 120, u * 9) - 0.5) * 0.2;
      const a2 = a + side * 0.9 * (u - 0.5) + warp * (0.4 + 0.6 * u);
      const w = sstep(-0.14, 0.14, side * Math.cos(a2));
      const k = j * NT + i;
      for (let c = 0; c < 3; c++) out[c][k] = A[c][k] * w + B[c][k] * (1 - w);
    }
  }
  return out;
}

function renderRows(M, S, pr, start, step, look) {
  // height map (for relief lighting) from luminance, and its derivatives
  const H = new Float32Array(N);
  for (let k = 0; k < N; k++) H[k] = Math.log(0.02 + lumOf(M[0][k], M[1][k], M[2][k]));
  const Hs = blur(H.slice(), 1, 1), Hu = new Float32Array(N), Ht = new Float32Array(N);
  for (let j = 0; j < NR; j++) for (let i = 0; i < NT; i++) {
    const k = j * NT + i;
    Ht[k] = (Hs[j * NT + (i + 1) % NT] - Hs[j * NT + (i - 1 + NT) % NT]) / 2;
    Hu[k] = (Hs[Math.min(NR - 1, j + 1) * NT + i] - Hs[Math.max(0, j - 1) * NT + i]) / 2;
  }
  const field = (arr, fi, fj) => {
    const i0 = fi | 0, i1 = (i0 + 1) % NT, ft = fi - i0, j0 = fj | 0, j1 = Math.min(NR - 1, j0 + 1), fr = fj - j0;
    const a = arr[j0 * NT + i0] + (arr[j0 * NT + i1] - arr[j0 * NT + i0]) * ft;
    const b = arr[j1 * NT + i0] + (arr[j1 * NT + i1] - arr[j1 * NT + i0]) * ft;
    return a + (b - a) * fr;
  };
  const C = S / 2, Ro = S * 0.41, ro = Ro * pr, span = Ro - ro;
  const Lx = -0.62, Ly = -0.78;                                  // light from the top left
  // ring light reflection (the "viral macro" look) or a soft window
  const rcx = C - Ro * 0.05, rcy = C - Ro * 0.07, rr = Ro * 0.4, rw = Ro * 0.022;
  const rows = Math.ceil((S - start) / step), out = new Uint8ClampedArray(rows * S * 4);
  let rnd = ((start + 1) * 2654435761) >>> 0 || 1;
  for (let y = start, row = 0; y < S; y += step, row++) {
    for (let x = 0; x < S; x++) {
      const dx = x + 0.5 - C, dy = y + 0.5 - C, d = Math.sqrt(dx * dx + dy * dy);
      let R_ = 0, G_ = 0, B_ = 0;
      // faint glow of the eye's white just outside the iris
      if (d >= Ro && d < Ro * 1.25) { const g = 0.05 * Math.exp(-(d - Ro) / (Ro * 0.05)); R_ = g * 0.95; G_ = g * 0.9; B_ = g * 0.92; }
      if (d < Ro * 1.03) {
        const ca = d > 0 ? dx / d : 1, sa = d > 0 ? dy / d : 0;
        const t = (Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI);
        // slightly irregular limbus and pupil edges
        const edgeN = noise2(t * 40, 1.5) - 0.5;
        const roE = ro * (1 + edgeN * 0.02), RoE = Ro * (1 + edgeN * 0.012);
        const uc = clamp((d - roE) / (RoE - roE), 0, 1);
        let fi = t * NT - 0.5; if (fi < 0) fi += NT;
        const fj = clamp(uc * NR - 0.5, 0, NR - 1.001);
        let r0 = field(M[0], fi, fj), g0 = field(M[1], fi, fj), b0 = field(M[2], fi, fj);
        // relief lighting from the height map
        const hu = field(Hu, fi, fj) * NR / span, ht = field(Ht, fi, fj) * NT / (2 * Math.PI) / Math.max(d, 1);
        const gx = hu * ca - ht * sa, gy = hu * sa + ht * ca;
        const shade = clamp(1 - 9 * (gx * Lx + gy * Ly) * (S / 1024), 0.6, 1.5);
        // micro fibres finer than the texture
        const mf = 1 + (noise2(t * 2900, uc * 22) - 0.5) * 0.09 + (noise2(t * 6200, uc * 60 + 5) - 0.5) * 0.06;
        const L = shade * mf;
        r0 *= L; g0 *= L; b0 *= L;
        // dark limbal ring and pupil ruff
        const limb = 1 - 0.72 * Math.pow(sstep(0.74, 1.0, uc), 1.3);
        r0 *= limb; g0 *= limb; b0 *= limb * 1.03;
        const ruff = sstep(0.045, 0.0, uc) * 0.7;
        r0 += (0.1 - r0) * ruff; g0 += (0.055 - g0) * ruff; b0 += (0.03 - b0) * ruff;
        // gentle colour grade: saturation + S-curve
        const gy0 = lumOf(r0, g0, b0);
        r0 = gy0 + (r0 - gy0) * 1.18; g0 = gy0 + (g0 - gy0) * 1.18; b0 = gy0 + (b0 - gy0) * 1.18;
        const curve = v => { v = clamp(v, 0, 1.4); return v < 1 ? v * v * (3 - 2 * v) * 0.35 + v * 0.65 : v; };
        r0 = curve(r0); g0 = curve(g0); b0 = curve(b0);
        const pin = sstep(roE - 1.2, roE + 1.2, d);
        const pupil = 0.006 + 0.012 * (d / ro) * (d / ro);
        const outE = sstep(RoE + S * 0.006, RoE - S * 0.002, d);
        R_ = Math.max(R_, (pupil + (r0 - pupil) * pin) * outE); G_ = Math.max(G_, (pupil + (g0 - pupil) * pin) * outE); B_ = Math.max(B_, (pupil + (b0 - pupil) * pin) * outE);
        // reflections on the cornea (screen blend), stronger over the dark pupil
        let cl = 0;
        if (look !== "plain") {
          const dr = Math.hypot(x - rcx, y - rcy), ra = Math.atan2(y - rcy, x - rcx);
          const gap = sstep(0.25, 0.4, noise2(ra * 2.2 + 10, 2.5));
          cl += Math.exp(-(((dr - rr) / rw) ** 2)) * 0.42 * gap;
          const wx = (x - (C - Ro * 0.36)) / (Ro * 0.11), wy = (y - (C - Ro * 0.42)) / (Ro * 0.07);
          cl += sstep(1, 0.55, Math.hypot(wx, wy)) * 0.45;
          cl *= 0.7 + 0.5 * (1 - pin);
        }
        cl += 0.05 * sstep(Ro * 1.1, 0, Math.hypot(x - (C - Ro * 0.5), y - (C - Ro * 0.55)));   // corneal sheen
        cl = Math.min(1, cl) * outE;
        R_ = 1 - (1 - R_) * (1 - cl); G_ = 1 - (1 - G_) * (1 - cl); B_ = 1 - (1 - B_) * (1 - cl);
      }
      rnd ^= rnd << 13; rnd ^= rnd >>> 17; rnd ^= rnd << 5;
      const gr = ((rnd >>> 0) / 4294967296 - 0.5) * 0.022;          // film grain
      const k = (row * S + x) * 4;
      out[k] = (R_ + gr) * 255; out[k + 1] = (G_ + gr) * 255; out[k + 2] = (B_ + gr) * 255; out[k + 3] = 255;
    }
  }
  return out;
}

self.onmessage = (e) => {
  // rows are interleaved across workers (start, start+step, ...) so every worker gets an equal share of iris
  const { id, A, B, libA, libB, swap, seed, S, start, step, look } = e.data;
  seedInt = seed | 0;
  let M = material(userFields(A.img, A.cx, A.cy, A.R, A.pr), libA), pr = A.pr;
  if (B) {
    M = fuse(M, material(userFields(B.img, B.cx, B.cy, B.R, B.pr), libB || libA), swap);
    pr = (A.pr + B.pr) / 2;
  }
  const band = renderRows(M, S, clamp(pr, 0.22, 0.5), start, step, look);
  self.postMessage({ id, start, step, band }, [band.buffer]);
};
