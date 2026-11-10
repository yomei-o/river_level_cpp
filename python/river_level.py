"""river_level.py — river_level.hpp / main.cpp の Python 版（同じ方法・同じオプション・同じ CSV）

mp4 から水面の y 座標を 7 通りの方法で求める。計算は C++ 版と同じ順序・同じ式で行い、
同じ画素からは同じ値が出るようにしてある（整数で済む和は整数で、浮動小数の和は C++ と同じ順に足す）。

    python river_level.py <入力.mp4> [--roi x0 y0 x1 y1] [--gauge gx0 gx1] [--csv out.csv]
                          [--overlay out.mp4] [--snap out.bmp] [--every N] [--max N]

必要なもの: numpy, av (PyAV)
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

NAN = float("nan")


@dataclass
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int


# ---------------------------------------------------------------------------
# 共通の道具
# ---------------------------------------------------------------------------
def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB(H,W,3 uint8) → グレー。C++ と同じ (77R + 150G + 29B) >> 8。"""
    c = rgb.astype(np.int32)
    return ((77 * c[..., 0] + 150 * c[..., 1] + 29 * c[..., 2]) >> 8).astype(np.uint8)


def row_mean(g: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    # 画素は整数なので和は厳密。C++ と同じく和 / 幅。
    return g[y0:y1, x0:x1].sum(axis=1, dtype=np.int64).astype(np.float64) / (x1 - x0)


def median(v) -> float:
    if len(v) == 0:
        return NAN
    return float(np.median(np.asarray(v, dtype=np.float64)))


def changepoint(p: np.ndarray, sign: int = 0, min_seg: int = 4) -> int:
    """p を 2 つの定数で近似したとき二乗誤差が最小の分け目 s（p[:s] | p[s:]）。無ければ -1。"""
    n = len(p)
    S = np.concatenate(([0.0], np.cumsum(p)))
    Q = np.concatenate(([0.0], np.cumsum(p * p)))
    s = np.arange(min_seg, n - min_seg + 1)
    if len(s) == 0:
        return -1
    ma = (S[s] - S[0]) / s
    mb = (S[n] - S[s]) / (n - s)
    sa = S[s] - S[0]
    sb = S[n] - S[s]
    v = ((Q[s] - Q[0]) - sa * sa / s) + ((Q[n] - Q[s]) - sb * sb / (n - s))
    ok = np.ones(len(s), bool)
    if sign > 0:
        ok = ma > mb
    elif sign < 0:
        ok = ma < mb
    if not ok.any():
        return -1
    v = np.where(ok, v, np.inf)
    return int(s[int(np.argmin(v))])


def strips(r: Roi, width: int):
    out = []
    x = r.x0
    while x + width // 2 < r.x1:
        out.append((x, min(x + width, r.x1)))
        x += width
    return out


def _step_profile(p: np.ndarray, h: int, divide: bool) -> np.ndarray:
    """各 y（h..n-h）で「上 h 行の和 − 下 h 行の和」。C++ と同じく k=1..h の順に足す。"""
    n = len(p)
    ys = np.arange(h, n - h + 1)
    a = np.zeros(len(ys))
    b = np.zeros(len(ys))
    for k in range(1, h + 1):
        a = a + p[ys - k]
        b = b + p[ys + k - 1]
    return (a - b) / h if divide else a - b


# ---------------------------------------------------------------------------
# 方法 A: 横エッジ（局所的な明→暗の段差）
# ---------------------------------------------------------------------------
def method_edge(g, r: Roi, strip_w: int, h: int, per_strip=None) -> float:
    ys = []
    for x0, x1 in strips(r, strip_w):
        d = _step_profile(row_mean(g, x0, x1, r.y0, r.y1), h, True)
        ys.append(float(r.y0 + h + int(np.argmax(d))) if len(d) else float(r.y0 - 1))
    if per_strip is not None:
        per_strip[:] = ys
    return median(ys)


# ---------------------------------------------------------------------------
# 方法 B: 明るさの 2 段近似（変化点検出）
# ---------------------------------------------------------------------------
def method_step(g, r: Roi, strip_w: int, per_strip=None) -> float:
    ys = [float(r.y0 + changepoint(row_mean(g, x0, x1, r.y0, r.y1), +1)) for x0, x1 in strips(r, strip_w)]
    if per_strip is not None:
        per_strip[:] = ys
    return median(ys)


# ---------------------------------------------------------------------------
# 方法 C: 鏡映対称（水面を境に上下が鏡写し）。正規化相関が最大の y。
# ---------------------------------------------------------------------------
def method_mirror(g, r: Roi, strip_w: int, K: int, per_strip=None) -> float:
    # 行どうしの内積 P[i,j] を一度に作り、y ごとの和は行和の区間和と P の反対角の和で求める。
    # どれも整数の和なので C++ の逐次和と完全に同じ値になる。
    ys = []
    for x0, x1 in strips(r, strip_w):
        R = g[r.y0:r.y1, x0:x1].astype(np.int64)
        nr = R.shape[0]
        js = np.arange(K, nr - K + 1)  # y - r.y0
        if len(js) == 0:
            ys.append(-1.0)
            continue
        rs = np.concatenate(([0], np.cumsum(R.sum(axis=1))))
        rq = np.concatenate(([0], np.cumsum((R * R).sum(axis=1))))
        P = R @ R.T
        sa = (rs[js] - rs[js - K]).astype(np.float64)
        sb = (rs[js + K] - rs[js]).astype(np.float64)
        saa = (rq[js] - rq[js - K]).astype(np.float64)
        sbb = (rq[js + K] - rq[js]).astype(np.float64)
        sab = np.zeros(len(js), np.int64)
        for k in range(K):
            sab += P[js - 1 - k, js + k]
        sab = sab.astype(np.float64)
        n = K * (x1 - x0)
        ca = saa - sa * sa / n
        cb = sbb - sb * sb / n
        cab = sab - sa * sb / n
        ok = (ca > 1e-9) & (cb > 1e-9)
        v = np.where(ok, cab / np.sqrt(np.where(ok, ca * cb, 1.0)), -1.0)
        ys.append(float(r.y0 + js[int(np.argmax(v))]))
    if per_strip is not None:
        per_strip[:] = ys
    return median(ys)


# ---------------------------------------------------------------------------
# 方法 D: 水際のちらつき（直近 T フレームの画素ごとの標準偏差が最大の行）
# ---------------------------------------------------------------------------
class TemporalVar:
    def __init__(self, T: int, r: Roi):
        self.T, self.r = T, r
        self.buf: deque = deque()

    def push(self, g):
        r = self.r
        self.buf.append(g[r.y0:r.y1, r.x0:r.x1].copy())
        if len(self.buf) > self.T:
            self.buf.popleft()

    def ready(self) -> bool:
        return len(self.buf) >= self.T

    def row_std(self, x0: int, x1: int) -> np.ndarray:
        r = self.r
        F = np.stack(self.buf).astype(np.int64)[:, :, x0 - r.x0:x1 - r.x0]  # (n, h, w)
        n = F.shape[0]
        s = F.sum(axis=0).astype(np.float64)
        q = (F * F).sum(axis=0).astype(np.float64)
        var = q / n - (s / n) * (s / n)
        sd = np.sqrt(np.maximum(0.0, var))
        acc = np.cumsum(sd, axis=1)[:, -1]  # C++ と同じく x の順に足す
        return acc / (x1 - x0)

    def estimate(self, strip_w: int, per_strip=None) -> float:
        ys = []
        for x0, x1 in strips(self.r, strip_w):
            p = self.row_std(x0, x1)
            v = (p[:-2] + p[1:-1]) + p[2:]  # y = 1 .. n-2
            ys.append(float(self.r.y0 + 1 + int(np.argmax(v))) if len(v) else float(self.r.y0))
        if per_strip is not None:
            per_strip[:] = ys
        return median(ys)


# ---------------------------------------------------------------------------
# 方法 E: 横線の直線当てはめ（RANSAC、乱数は C++ と同じ xorshift32）
# ---------------------------------------------------------------------------
@dataclass
class Line:
    a: float = 0.0
    b: float = NAN
    inliers: int = 0

    def at(self, x: float) -> float:
        return self.a * x + self.b


def method_line(g, r: Roi, h: int, tol: float = 2.5, iters: int = 300, seed: int = 1) -> Line:
    pts = []
    cw = 5
    x0 = r.x0
    while x0 + cw <= r.x1:
        d = _step_profile(row_mean(g, x0, x0 + cw, r.y0, r.y1), h, False)
        by = h + int(np.argmax(d)) if len(d) else -1
        pts.append((x0 + cw / 2.0, float(r.y0 + by)))
        x0 += cw
    best = Line()
    if len(pts) < 2:
        return best
    st = seed if seed else 1
    M = 0xFFFFFFFF

    def U():
        nonlocal st
        st ^= (st << 13) & M
        st ^= st >> 17
        st ^= (st << 5) & M
        return st % len(pts)

    for _ in range(iters):
        p = pts[U()]
        q = pts[U()]
        if abs(p[0] - q[0]) < 30:
            continue
        a = (q[1] - p[1]) / (q[0] - p[0])
        if abs(a) > 0.2:
            continue
        b = p[1] - a * p[0]
        c = sum(1 for t in pts if abs(a * t[0] + b - t[1]) < tol)
        if c > best.inliers:
            best = Line(a, b, c)
    sx = sy = sxx = sxy = 0.0
    n = 0
    for t in pts:
        if abs(best.at(t[0]) - t[1]) < tol:
            sx += t[0]; sy += t[1]; sxx += t[0] * t[0]; sxy += t[0] * t[1]; n += 1
    if n >= 2:
        d = n * sxx - sx * sx
        if abs(d) > 1e-9:
            best.a = (n * sxy - sx * sy) / d
            best.b = (sy - best.a * sx) / n
        best.inliers = n
    return best


# ---------------------------------------------------------------------------
# 色の暖かさ R−B と、方法 F（水位標の下端）・G（色の 2 段近似）
# ---------------------------------------------------------------------------
def row_warmth(rgb, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    c = rgb[y0:y1, x0:x1].astype(np.int64)
    return (c[..., 0] - c[..., 2]).sum(axis=1).astype(np.float64) / (x1 - x0)


def method_gauge(rgb, gx0: int, gx1: int, y0: int, y1: int) -> float:
    return float(y0 + changepoint(row_warmth(rgb, gx0, gx1, y0, y1), +1))


def method_color(rgb, r: Roi, strip_w: int, per_strip=None) -> float:
    ys = [float(r.y0 + changepoint(row_warmth(rgb, x0, x1, r.y0, r.y1), +1)) for x0, x1 in strips(r, strip_w)]
    if per_strip is not None:
        per_strip[:] = ys
    return median(ys)


# ---------------------------------------------------------------------------
# 描画（オーバーレイ）
# ---------------------------------------------------------------------------
C_EDGE, C_STEP, C_MIRROR, C_TEMP = (255, 60, 60), (255, 200, 0), (0, 220, 255), (180, 80, 255)
C_LINE, C_GAUGE, C_COLOR = (60, 255, 60), (255, 255, 255), (255, 120, 200)


def _hline(im, x0, x1, y, c):
    if math.isnan(y):
        return
    yi = int(math.floor(y + 0.5)) if y >= 0 else -int(math.floor(-y + 0.5))  # std::lround と同じ
    if 0 <= yi < im.shape[0]:
        im[yi, max(0, x0):max(0, min(x1, im.shape[1]))] = c


def _seg(im, x0, x1, L: Line, c):
    if math.isnan(L.b):
        return
    for x in range(x0, x1):
        y = L.at(x)
        yi = int(math.floor(y + 0.5)) if y >= 0 else -int(math.floor(-y + 0.5))
        if 0 <= yi < im.shape[0] and 0 <= x < im.shape[1]:
            im[yi, x] = c


def _rect(im, r: Roi, c):
    for x in range(r.x0, r.x1, 2):
        im[r.y0, x] = c; im[r.y1 - 1, x] = c
    for y in range(r.y0, r.y1, 2):
        im[y, r.x0] = c; im[y, r.x1 - 1] = c


def draw_overlay(rgb, roi, strip_w, se, ss, sm, st, sc, L, gx0, gx1, yg):
    o = rgb.copy()
    _rect(o, roi, (255, 255, 255))
    for i, (x0, x1) in enumerate(strips(roi, strip_w)):
        q = (x1 - x0) // 5
        _hline(o, x0, x0 + q, se[i], C_EDGE)
        _hline(o, x0 + q, x0 + 2 * q, ss[i], C_STEP)
        _hline(o, x0 + 2 * q, x0 + 3 * q, sm[i], C_MIRROR)
        if i < len(st):
            _hline(o, x0 + 3 * q, x0 + 4 * q, st[i], C_TEMP)
        _hline(o, x0 + 4 * q, x1, sc[i], C_COLOR)
    _seg(o, 0, o.shape[1], L, C_LINE)
    _hline(o, gx0 - 15, gx1 + 15, yg, C_GAUGE)
    _hline(o, gx0 - 15, gx1 + 15, yg + 1, C_GAUGE)
    for k, c in enumerate((C_EDGE, C_STEP, C_MIRROR, C_TEMP, C_LINE, C_GAUGE, C_COLOR)):
        for dy in range(4):
            _hline(o, 20 + k * 40, 20 + k * 40 + 30, 640 + dy, c)
    return o


def write_bmp(path: str, im: np.ndarray):
    h, w, _ = im.shape
    row = (w * 3 + 3) & ~3
    size = 54 + row * h
    with open(path, "wb") as f:
        f.write(b"BM" + struct.pack("<IHHI", size, 0, 0, 54))
        f.write(struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, size - 54, 0, 0, 0, 0))
        pad = b"\0" * (row - w * 3)
        for y in range(h - 1, -1, -1):
            f.write(im[y, :, ::-1].tobytes() + pad)


# ---------------------------------------------------------------------------
# コマンド（main.cpp と同じオプション・同じ出力）
# ---------------------------------------------------------------------------
def main(argv=None):
    import av

    ap = argparse.ArgumentParser(description="川の水面の y 座標を 7 通りの方法で求める")
    ap.add_argument("input")
    ap.add_argument("--roi", nargs=4, type=int, default=[300, 300, 1150, 520], metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--gauge", nargs=2, type=int, default=[621, 630], metavar=("GX0", "GX1"))
    ap.add_argument("--csv", default="levels.csv")
    ap.add_argument("--overlay", default="")
    ap.add_argument("--snap", default="")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--strip", type=int, default=85)
    ap.add_argument("--edge-h", type=int, default=4)
    ap.add_argument("--mirror-k", type=int, default=20)
    ap.add_argument("--temporal-t", type=int, default=30)
    a = ap.parse_args(argv)

    roi = Roi(*a.roi)
    gx0, gx1 = a.gauge
    W, H = 1280, 720
    # .rgb は生の RGB24 1280x720 30fps（同じ画素で C++ 版と比べる検証用）、それ以外は mp4
    raw = a.input.endswith(".rgb")
    if raw:
        src = None
        fps, in_w, in_h = 30.0, W, H

        def frames():
            with open(a.input, "rb") as f:
                while True:
                    b = f.read(W * H * 3)
                    if len(b) < W * H * 3:
                        return
                    yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    else:
        src = av.open(a.input)
        vs = src.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 30.0
        in_w, in_h = vs.codec_context.width, vs.codec_context.height

        def frames():
            for fr in src.decode(vs):
                yield fr.to_ndarray(format="rgb24", width=W, height=H)

    out = ovs = None
    if a.overlay:
        out = av.open(a.overlay, "w")
        ovs = out.add_stream("libx264" if "libx264" in av.codecs_available else "h264", rate=max(1, round(fps / a.every)))
        ovs.width, ovs.height, ovs.pix_fmt = W, H, "yuv420p"
        ovs.bit_rate = 4_000_000

    csv = open(a.csv, "w", newline="\n")
    csv.write("frame,edge,step,mirror,temporal,line_at_gauge,line_slope,line_inliers,gauge,color\n")
    tv = TemporalVar(a.temporal_t, roi)
    gauge_cx = (gx0 + gx1) / 2.0
    hist = {k: [] for k in ("A edge    ", "B step    ", "C mirror  ", "D temporal", "E line    ", "F gauge   ", "G color   ")}
    used = 0
    snapped = False

    def f1(v):
        return "nan" if math.isnan(v) else f"{v:.1f}"

    for fi, rgb in enumerate(frames()):
        if fi % a.every:
            continue
        g = to_gray(rgb)
        se, ss, sm, st, sc = [], [], [], [], []
        ye = method_edge(g, roi, a.strip, a.edge_h, se)
        ys = method_step(g, roi, a.strip, ss)
        ym = method_mirror(g, roi, a.strip, a.mirror_k, sm)
        tv.push(g)
        yt = tv.estimate(a.strip, st) if tv.ready() else NAN
        L = method_line(g, roi, a.edge_h)
        yg = method_gauge(rgb, gx0, gx1, roi.y0, roi.y1)
        yc = method_color(rgb, roi, a.strip, sc)

        csv.write(f"{fi},{f1(ye)},{f1(ys)},{f1(ym)},{f1(yt)},{L.at(gauge_cx):.2f},{L.a:.5f},{L.inliers},{f1(yg)},{f1(yc)}\n")
        hist["A edge    "].append(ye)
        hist["B step    "].append(ys)
        hist["C mirror  "].append(ym)
        if not math.isnan(yt):
            hist["D temporal"].append(yt)
        hist["E line    "].append(L.at(gauge_cx))
        hist["F gauge   "].append(yg)
        hist["G color   "].append(yc)

        want_snap = bool(a.snap) and not snapped and tv.ready()
        if out or want_snap:
            o = draw_overlay(rgb, roi, a.strip, se, ss, sm, st, sc, L, gx0, gx1, yg)
            if out:
                vf = av.VideoFrame.from_ndarray(o, format="rgb24")
                for pkt in ovs.encode(vf):
                    out.mux(pkt)
            if want_snap:
                write_bmp(a.snap, o)
                snapped = True
        used += 1
        if a.max and used >= a.max:
            break

    if out:
        for pkt in ovs.encode():
            out.mux(pkt)
        out.close()
    if src:
        src.close()
    csv.close()

    print(f"{a.input}: {in_w}x{in_h} {fps:.2f} fps, frames analysed: {used} (every {a.every})")
    print("method       median     p10     p90    (y, px; smaller = higher water)")
    for k, v in sorted(hist.items()):
        if not v:
            continue
        s = sorted(v)
        q = lambda f: s[int(min(len(s) - 1, f * (len(s) - 1)))]
        print(f"{k}  {median(v):7.1f} {q(0.1):7.1f} {q(0.9):7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
