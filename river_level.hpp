// river_level.hpp — 川の水面の y 座標を映像から求める（ヘッダーオンリー、依存なし）
//
// 入力はグレースケール画像（1 byte/pixel）。ROI の中だけを見て、
// 「水面＝護岸（上）と水（下）の境目の行」を何種類かの方法で推定する。
// 返す y は全画面座標で、「水になる最初の行」。
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

namespace rl {

struct Gray {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
    uint8_t at(int x, int y) const { return px[(size_t)y * w + x]; }
};

struct Roi { int x0, y0, x1, y1; };  // [x0,x1) × [y0,y1)

// 行ごとの平均輝度（x0..x1、y0..y1）
inline std::vector<double> row_mean(const Gray& g, int x0, int x1, int y0, int y1) {
    std::vector<double> p(y1 - y0, 0.0);
    for (int y = y0; y < y1; ++y) {
        double s = 0;
        const uint8_t* r = &g.px[(size_t)y * g.w];
        for (int x = x0; x < x1; ++x) s += r[x];
        p[y - y0] = s / (x1 - x0);
    }
    return p;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    double m = v[n];
    if (v.size() % 2 == 0) m = (m + *std::max_element(v.begin(), v.begin() + n)) / 2;
    return m;
}

// 1 本の列 profile を「上の段」と「下の段」の 2 つの定数で近似したとき
// 二乗誤差が最小になる分け目 s（p[0..s) | p[s..n)）。
// sign=+1 なら「上が明るい」分け目だけを候補にする。
inline int changepoint(const std::vector<double>& p, int sign = 0, int min_seg = 4) {
    int n = (int)p.size();
    std::vector<double> S(n + 1, 0), Q(n + 1, 0);
    for (int i = 0; i < n; ++i) { S[i + 1] = S[i] + p[i]; Q[i + 1] = Q[i] + p[i] * p[i]; }
    auto sse = [&](int a, int b) { double s = S[b] - S[a]; return Q[b] - Q[a] - s * s / (b - a); };
    int best = -1; double bv = 1e300;
    for (int s = min_seg; s <= n - min_seg; ++s) {
        double ma = (S[s] - S[0]) / s, mb = (S[n] - S[s]) / (n - s);
        if (sign > 0 && ma <= mb) continue;
        if (sign < 0 && ma >= mb) continue;
        double v = sse(0, s) + sse(s, n);
        if (v < bv) { bv = v; best = s; }
    }
    return best;
}

// ROI を縦の短冊に分けたときの各短冊の x 範囲
inline std::vector<std::pair<int, int>> strips(const Roi& r, int width) {
    std::vector<std::pair<int, int>> v;
    for (int x = r.x0; x + width / 2 < r.x1; x += width) v.push_back({x, std::min(x + width, r.x1)});
    return v;
}

// ---------------------------------------------------------------------------
// 方法 A: 横エッジ（局所的な明→暗の段差）
//   各行 y で「上 h 行の平均 − 下 h 行の平均」を計算し、最大の行を水面とする。
//   石積みの目地（細い暗線）は h を数行にすると打ち消し合い、
//   壁→水の大きな段差だけが残る。短冊ごとに求めて中央値。
// ---------------------------------------------------------------------------
inline double method_edge(const Gray& g, const Roi& r, int strip_w, int h, std::vector<double>* per_strip = nullptr) {
    std::vector<double> ys;
    for (auto [x0, x1] : strips(r, strip_w)) {
        auto p = row_mean(g, x0, x1, r.y0, r.y1);
        int n = (int)p.size(); int by = -1; double bv = -1e9;
        for (int y = h; y + h <= n; ++y) {
            double a = 0, b = 0;
            for (int k = 1; k <= h; ++k) { a += p[y - k]; b += p[y + k - 1]; }
            double d = (a - b) / h;
            if (d > bv) { bv = d; by = y; }
        }
        ys.push_back(r.y0 + by);
    }
    if (per_strip) *per_strip = ys;
    return median(ys);
}

// ---------------------------------------------------------------------------
// 方法 B: 2 段近似（変化点検出）
//   行平均の列を「明るい上段」「暗い下段」の 2 つの定数で近似し、
//   誤差最小の分け目を水面とする。局所の段差ではなく ROI 全体の明暗で決まる。
// ---------------------------------------------------------------------------
inline double method_step(const Gray& g, const Roi& r, int strip_w, std::vector<double>* per_strip = nullptr) {
    std::vector<double> ys;
    for (auto [x0, x1] : strips(r, strip_w)) {
        auto p = row_mean(g, x0, x1, r.y0, r.y1);
        ys.push_back(r.y0 + changepoint(p, +1));
    }
    if (per_strip) *per_strip = ys;
    return median(ys);
}

// ---------------------------------------------------------------------------
// 方法 C: 鏡映対称（映り込み）
//   水面は鏡なので、水面 y を境に「上の y-1-k 行」と「下の y+k 行」が似る。
//   映り込みは暗く低コントラストなので、明るさと倍率に依らない正規化相関(NCC)で比べ、
//   相関が最大の y を水面とする。
// ---------------------------------------------------------------------------
inline double method_mirror(const Gray& g, const Roi& r, int strip_w, int K, std::vector<double>* per_strip = nullptr) {
    std::vector<double> ys;
    for (auto [x0, x1] : strips(r, strip_w)) {
        int by = -1; double bv = -2;
        for (int y = r.y0 + K; y + K <= r.y1; ++y) {
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0; int n = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t* ra = &g.px[(size_t)(y - 1 - k) * g.w];
                const uint8_t* rb = &g.px[(size_t)(y + k) * g.w];
                for (int x = x0; x < x1; ++x) {
                    double a = ra[x], b = rb[x];
                    sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; ++n;
                }
            }
            double ca = saa - sa * sa / n, cb = sbb - sb * sb / n, cab = sab - sa * sb / n;
            double v = (ca > 1e-9 && cb > 1e-9) ? cab / std::sqrt(ca * cb) : -1;
            if (v > bv) { bv = v; by = y; }
        }
        ys.push_back(by);
    }
    if (per_strip) *per_strip = ys;
    return median(ys);
}

// ---------------------------------------------------------------------------
// 方法 D: 時間変動（水際のちらつき）
//   護岸も、穏やかな水面もほとんど動かない。動くのは護岸に当たる水際の細い帯だけ
//   （小さな波が石を濡らしたり乾かしたりする）。直近 T フレームの画素ごとの
//   標準偏差を行平均し、最大の行を水面とする。1 枚では決まらず、動画が要る。
// ---------------------------------------------------------------------------
struct TemporalVar {
    int T;
    Roi r;
    std::deque<std::vector<uint8_t>> buf;
    TemporalVar(int T_, Roi r_) : T(T_), r(r_) {}
    void push(const Gray& g) {
        int w = r.x1 - r.x0, h = r.y1 - r.y0;
        std::vector<uint8_t> c((size_t)w * h);
        for (int y = 0; y < h; ++y)
            std::copy_n(&g.px[(size_t)(r.y0 + y) * g.w + r.x0], w, &c[(size_t)y * w]);
        buf.push_back(std::move(c));
        if ((int)buf.size() > T) buf.pop_front();
    }
    bool ready() const { return (int)buf.size() >= T; }
    // 行ごとの平均標準偏差（可視化にも使う）
    std::vector<double> row_std(int x0, int x1) const {
        int w = r.x1 - r.x0, h = r.y1 - r.y0, n = (int)buf.size();
        std::vector<double> p(h, 0);
        for (int y = 0; y < h; ++y) {
            double acc = 0;
            for (int x = x0 - r.x0; x < x1 - r.x0; ++x) {
                double s = 0, q = 0;
                for (auto& f : buf) { double v = f[(size_t)y * w + x]; s += v; q += v * v; }
                double var = q / n - (s / n) * (s / n);
                acc += std::sqrt(std::max(0.0, var));
            }
            p[y] = acc / (x1 - x0);
        }
        return p;
    }
    double estimate(int strip_w, std::vector<double>* per_strip = nullptr) const {
        std::vector<double> ys;
        for (auto [x0, x1] : strips(r, strip_w)) {
            auto p = row_std(x0, x1);
            // 3 行の移動平均で圧縮ノイズを均してから最大を取る
            int by = 0; double bv = -1;
            for (int y = 1; y + 1 < (int)p.size(); ++y) {
                double v = p[y - 1] + p[y] + p[y + 1];
                if (v > bv) { bv = v; by = y; }
            }
            ys.push_back(r.y0 + by);
        }
        if (per_strip) *per_strip = ys;
        return median(ys);
    }
};

// ---------------------------------------------------------------------------
// 方法 E: 横線の直線当てはめ（RANSAC）
//   列ごとに「明→暗の段差が最大の行」を 1 点ずつ拾い、その点群に直線 y = a x + b を
//   RANSAC で当てはめる。草・水位標・葉で外れた列は外れ値として捨てられる。
//   カメラが少し傾いていても線の傾きごと求まる。
// ---------------------------------------------------------------------------
struct Line { double a = 0, b = NAN; int inliers = 0; double at(double x) const { return a * x + b; } };

inline Line method_line(const Gray& g, const Roi& r, int h, double tol = 2.5, int iters = 300, uint32_t seed = 1) {
    // 列方向に 5 画素ずつまとめて点を作る（ノイズ低減）
    std::vector<std::pair<double, double>> pts;
    const int cw = 5;
    for (int x0 = r.x0; x0 + cw <= r.x1; x0 += cw) {
        auto p = row_mean(g, x0, x0 + cw, r.y0, r.y1);
        int n = (int)p.size(); int by = -1; double bv = -1e9;
        for (int y = h; y + h <= n; ++y) {
            double a = 0, b = 0;
            for (int k = 1; k <= h; ++k) { a += p[y - k]; b += p[y + k - 1]; }
            if (a - b > bv) { bv = a - b; by = y; }
        }
        pts.push_back({x0 + cw / 2.0, (double)(r.y0 + by)});
    }
    Line best;
    if (pts.size() < 2) return best;
    // 乱数は xorshift32 を自前で（std の分布は実装ごとに違い、Python 版と同じ点を選べないため）
    uint32_t st = seed ? seed : 1;
    auto U = [&]() -> size_t { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st % pts.size(); };
    for (int it = 0; it < iters; ++it) {
        auto p = pts[U()];
        auto q = pts[U()];
        if (std::fabs(p.first - q.first) < 30) continue;
        double a = (q.second - p.second) / (q.first - p.first);
        if (std::fabs(a) > 0.2) continue;  // ほぼ水平な線だけ
        double b = p.second - a * p.first;
        int c = 0;
        for (auto& t : pts) if (std::fabs(a * t.first + b - t.second) < tol) ++c;
        if (c > best.inliers) { best.a = a; best.b = b; best.inliers = c; }
    }
    // インライアで最小二乗をやり直す
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
    for (auto& t : pts) if (std::fabs(best.at(t.first) - t.second) < tol) {
        sx += t.first; sy += t.second; sxx += t.first * t.first; sxy += t.first * t.second; ++n;
    }
    if (n >= 2) {
        double d = n * sxx - sx * sx;
        if (std::fabs(d) > 1e-9) { best.a = (n * sxy - sx * sy) / d; best.b = (sy - best.a * sx) / n; }
        best.inliers = n;
    }
    return best;
}

// ---------------------------------------------------------------------------
// 色の暖かさ R−B（護岸・水位標は R>B、水は空を映して B>R）
// ---------------------------------------------------------------------------
struct Rgb { int w = 0, h = 0; std::vector<uint8_t> px; };

inline std::vector<double> row_warmth(const Rgb& c, int x0, int x1, int y0, int y1) {
    std::vector<double> p(y1 - y0, 0.0);
    for (int y = y0; y < y1; ++y) {
        double s = 0;
        for (int x = x0; x < x1; ++x) { const uint8_t* q = &c.px[((size_t)y * c.w + x) * 3]; s += (double)q[0] - q[2]; }
        p[y - y0] = s / (x1 - x0);
    }
    return p;
}

// ---------------------------------------------------------------------------
// 方法 F: 水位標（量水標）の下端
//   水位標は白・橙で暖色。水中に続く部分は水を通して青みがかった灰色になる。
//   水位標の列だけの R−B を上から見て、「暖色の上段／青い下段」の変化点を水面とする。
// ---------------------------------------------------------------------------
inline double method_gauge(const Rgb& c, int gx0, int gx1, int y0, int y1, std::vector<double>* score = nullptr) {
    auto s = row_warmth(c, gx0, gx1, y0, y1);
    if (score) *score = s;
    return y0 + changepoint(s, +1);
}

// ---------------------------------------------------------------------------
// 方法 G: 色（R−B）の 2 段近似
//   F と同じ色の手がかりを護岸全体に使う。短冊ごとに R−B の行平均の変化点を取り、中央値。
//   明るさ（B）と違い、水際の濡れて暗い石の段も「護岸色」のままなので段差がずれにくい。
// ---------------------------------------------------------------------------
inline double method_color(const Rgb& c, const Roi& r, int strip_w, std::vector<double>* per_strip = nullptr) {
    std::vector<double> ys;
    for (auto [x0, x1] : strips(r, strip_w)) ys.push_back(r.y0 + changepoint(row_warmth(c, x0, x1, r.y0, r.y1), +1));
    if (per_strip) *per_strip = ys;
    return median(ys);
}

}  // namespace rl
