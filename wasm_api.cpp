// wasm_api.cpp — river_level.hpp をブラウザから呼ぶための C 関数（emscripten でビルド）
//
// JS 側は canvas の RGBA をそのまま rl_frame_buffer() に書き込み、rl_process() を呼ぶ。
// 結果は rl_results() が指す double 配列に入る（並びは下の RES_* を参照）。
#include <cstdint>
#include <memory>
#include <vector>

#include "river_level.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define API extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define API extern "C"
#endif

using namespace rl;

namespace {
int W = 0, H = 0;
std::vector<uint8_t> rgba;
Rgb rgb;
Gray gray;
Roi roi{300, 300, 1150, 520};
int gx0 = 621, gx1 = 630, strip_w = 85, edge_h = 4, mirror_k = 20, temporal_T = 30;
bool use_mirror = true;
std::unique_ptr<TemporalVar> tv;
std::vector<double> res;
}  // namespace

// 結果配列の並び
enum {
    RES_EDGE, RES_STEP, RES_MIRROR, RES_TEMPORAL, RES_LINE_A, RES_LINE_B, RES_LINE_INLIERS, RES_GAUGE, RES_COLOR,
    RES_NSTRIPS,  // 以降、短冊ごとに [x0, x1, edge, step, mirror, temporal, color] が並ぶ
    RES_HEAD
};

API uint8_t* rl_frame_buffer(int w, int h) {
    if (w != W || h != H) {
        W = w; H = h;
        rgba.assign((size_t)w * h * 4, 0);
        rgb.w = w; rgb.h = h; rgb.px.assign((size_t)w * h * 3, 0);
        gray.w = w; gray.h = h; gray.px.assign((size_t)w * h, 0);
        tv.reset();
    }
    return rgba.data();
}

API void rl_set_params(int x0, int y0, int x1, int y1, int g0, int g1, int strip, int eh, int mk, int T, int mirror) {
    roi = {x0, y0, x1, y1};
    gx0 = g0; gx1 = g1; strip_w = strip; edge_h = eh; mirror_k = mk; temporal_T = T; use_mirror = mirror != 0;
    tv.reset();  // ROI が変わったら時間変動の履歴は捨てる
}

API void rl_reset() { tv.reset(); }

API double* rl_results() { return res.data(); }

API int rl_process() {
    if (!W || roi.x1 > W || roi.y1 > H || roi.x0 < 0 || roi.y0 < 0 || roi.x1 - roi.x0 < 20 || roi.y1 - roi.y0 < 3 * mirror_k) return 0;
    for (size_t i = 0, n = (size_t)W * H; i < n; ++i) {
        const uint8_t* p = &rgba[i * 4];
        rgb.px[i * 3] = p[0]; rgb.px[i * 3 + 1] = p[1]; rgb.px[i * 3 + 2] = p[2];
        gray.px[i] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
    }
    if (!tv) tv = std::make_unique<TemporalVar>(temporal_T, roi);

    std::vector<double> se, ss, sm, st, sc;
    double ye = method_edge(gray, roi, strip_w, edge_h, &se);
    double ys = method_step(gray, roi, strip_w, &ss);
    double ym = use_mirror ? method_mirror(gray, roi, strip_w, mirror_k, &sm) : NAN;
    tv->push(gray);
    double yt = tv->ready() ? tv->estimate(strip_w, &st) : NAN;
    Line L = method_line(gray, roi, edge_h);
    int gy0 = roi.y0, gy1 = roi.y1;
    double yg = (gx0 >= 12 && gx1 + 12 <= W && gx1 > gx0) ? method_gauge(rgb, gx0, gx1, gy0, gy1) : NAN;
    double yc = method_color(rgb, roi, strip_w, &sc);

    auto sx = strips(roi, strip_w);
    res.assign(RES_HEAD + sx.size() * 7, NAN);
    res[RES_EDGE] = ye; res[RES_STEP] = ys; res[RES_MIRROR] = ym; res[RES_TEMPORAL] = yt;
    res[RES_LINE_A] = L.a; res[RES_LINE_B] = L.b; res[RES_LINE_INLIERS] = L.inliers;
    res[RES_GAUGE] = yg; res[RES_COLOR] = yc; res[RES_NSTRIPS] = (double)sx.size();
    for (size_t i = 0; i < sx.size(); ++i) {
        double* o = &res[RES_HEAD + i * 7];
        o[0] = sx[i].first; o[1] = sx[i].second;
        o[2] = se[i]; o[3] = ss[i]; o[4] = i < sm.size() ? sm[i] : NAN; o[5] = i < st.size() ? st[i] : NAN; o[6] = sc[i];
    }
    return 1;
}
