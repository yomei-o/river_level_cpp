// river_level — mp4 から水面の y 座標を何種類かの方法で求める
//
// mp4 の展開・RGB 変換・書き出しは mediakit_cpp（Mp4Reader / VideoDecoder / FrameConverter /
// VideoEncoder / Mp4Writer）。ライブ配信はまず record.py で mp4 に録ってから渡す。
// 使い方:
//   river_level <入力.mp4> [--roi x0 y0 x1 y1] [--gauge gx0 gx1] [--csv out.csv]
//               [--overlay out.mp4] [--snap out.bmp] [--every N] [--max N]
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "mk_codec.h"
#include "river_level.hpp"

using namespace rl;

struct Args {
    std::string in, csv = "levels.csv", overlay, snap;
    Roi roi{300, 300, 1150, 520};
    int gx0 = 621, gx1 = 630;  // 水位標の列
    int every = 1, max_frames = 0;
    int strip = 85, edge_h = 4, mirror_k = 20, temporal_T = 30;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); exit(2); } return std::string(argv[++i]); };
        if (s == "--roi") { a.roi.x0 = std::stoi(nx()); a.roi.y0 = std::stoi(nx()); a.roi.x1 = std::stoi(nx()); a.roi.y1 = std::stoi(nx()); }
        else if (s == "--gauge") { a.gx0 = std::stoi(nx()); a.gx1 = std::stoi(nx()); }
        else if (s == "--csv") a.csv = nx();
        else if (s == "--overlay") a.overlay = nx();
        else if (s == "--snap") a.snap = nx();
        else if (s == "--every") a.every = std::stoi(nx());
        else if (s == "--max") a.max_frames = std::stoi(nx());
        else if (s == "--strip") a.strip = std::stoi(nx());
        else if (s == "--edge-h") a.edge_h = std::stoi(nx());
        else if (s == "--mirror-k") a.mirror_k = std::stoi(nx());
        else if (s == "--temporal-t") a.temporal_T = std::stoi(nx());
        else if (s[0] != '-' && a.in.empty()) a.in = s;
        else { fprintf(stderr, "unknown option %s\n", s.c_str()); exit(2); }
    }
    if (a.in.empty()) {
        fprintf(stderr, "usage: river_level <video.mp4> [--roi x0 y0 x1 y1] [--gauge gx0 gx1] [--csv f] [--overlay f.mp4] [--snap f.bmp] [--every N] [--max N]\n");
        exit(2);
    }
    return a;
}

// ---- 描画（オーバーレイ用） ----
struct Color { uint8_t r, g, b; };
static const Color C_EDGE{255, 60, 60}, C_STEP{255, 200, 0}, C_MIRROR{0, 220, 255}, C_TEMP{180, 80, 255}, C_LINE{60, 255, 60},
    C_GAUGE{255, 255, 255}, C_COLOR{255, 120, 200};

static void put(Rgb& im, int x, int y, Color c) {
    if (x < 0 || y < 0 || x >= im.w || y >= im.h) return;
    uint8_t* p = &im.px[((size_t)y * im.w + x) * 3]; p[0] = c.r; p[1] = c.g; p[2] = c.b;
}
static void hline(Rgb& im, int x0, int x1, double y, Color c) {
    if (std::isnan(y)) return;
    int yi = (int)std::lround(y);
    for (int x = x0; x < x1; ++x) put(im, x, yi, c);
}
static void seg(Rgb& im, int x0, int x1, const Line& L, Color c) {
    if (std::isnan(L.b)) return;
    for (int x = x0; x < x1; ++x) put(im, x, (int)std::lround(L.at(x)), c);
}
static void rect(Rgb& im, const Roi& r, Color c) {
    for (int x = r.x0; x < r.x1; x += 2) { put(im, x, r.y0, c); put(im, x, r.y1 - 1, c); }
    for (int y = r.y0; y < r.y1; y += 2) { put(im, r.x0, y, c); put(im, r.x1 - 1, y, c); }
}
// 凡例の色見本（左から A B C D E F G）
static void legend(Rgb& im, int x, int y, Color c) { for (int dy = 0; dy < 4; ++dy) hline(im, x, x + 30, y + dy, c); }

static bool write_bmp(const std::string& path, const Rgb& im) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    int row = (im.w * 3 + 3) & ~3;
    uint32_t size = 54 + (uint32_t)row * im.h;
    uint8_t h[54] = {'B', 'M'};
    auto u32 = [&](int o, uint32_t v) { for (int i = 0; i < 4; ++i) h[o + i] = (uint8_t)(v >> (8 * i)); };
    u32(2, size); u32(10, 54); u32(14, 40); u32(18, im.w); u32(22, im.h);
    h[26] = 1; h[28] = 24; u32(34, size - 54);
    fwrite(h, 1, 54, f);
    std::vector<uint8_t> line(row, 0);
    for (int y = im.h - 1; y >= 0; --y) {
        for (int x = 0; x < im.w; ++x) {
            const uint8_t* p = &im.px[((size_t)y * im.w + x) * 3];
            line[x * 3] = p[2]; line[x * 3 + 1] = p[1]; line[x * 3 + 2] = p[0];
        }
        fwrite(line.data(), 1, row, f);
    }
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    Args a = parse(argc, argv);

    mk::Mp4Reader reader;
    if (!reader.open(a.in)) { fprintf(stderr, "open %s: %s\n", a.in.c_str(), reader.error().c_str()); return 1; }
    const mk::MediaInfo info = reader.info();
    if (!info.has_video) { fprintf(stderr, "%s: no video\n", a.in.c_str()); return 1; }
    mk::VideoDecoder dec;
    if (!dec.open(false)) { fprintf(stderr, "decoder: %s\n", dec.error().c_str()); return 1; }
    mk::FrameConverter conv;

    // 座標は 1280x720 に揃えた画面で数える（WASM デモと同じ）
    const int W = 1280, H = 720;
    const double fps = info.fps > 0 ? info.fps : 30;

    mk::VideoEncoder enc;
    mk::Mp4Writer writer;
    if (!a.overlay.empty()) {
        mk::VideoEncoderConfig ec;
        ec.width = W; ec.height = H; ec.fps = std::max(1, (int)std::lround(fps / a.every)); ec.bitrate = 4000000; ec.gop = ec.fps;
        if (!enc.open(ec)) { fprintf(stderr, "encoder: %s\n", enc.error().c_str()); return 1; }
        if (!writer.open(a.overlay, &enc, nullptr)) { fprintf(stderr, "writer: %s\n", writer.error().c_str()); return 1; }
    }

    FILE* csv = fopen(a.csv.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", a.csv.c_str()); return 1; }
    fprintf(csv, "frame,edge,step,mirror,temporal,line_at_gauge,line_slope,line_inliers,gauge,color\n");

    Rgb rgb; rgb.w = W; rgb.h = H; rgb.px.resize((size_t)W * H * 3);
    Gray g; g.w = W; g.h = H; g.px.resize((size_t)W * H);
    TemporalVar tv(a.temporal_T, a.roi);
    const double gauge_cx = (a.gx0 + a.gx1) / 2.0;
    std::map<std::string, std::vector<double>> hist;
    int frame = 0, used = 0;
    bool done = false, snapped = false;
    std::vector<mk::Packet> enc_out;

    auto handle = [&](const mk::VideoFrame& vf) {
        int fi = frame++;
        if (done || fi % a.every) return;
        mk::VideoFrame rf;
        if (!conv.convert(vf, rf, mk::PixelFormat::RGB24, W, H)) { fprintf(stderr, "convert failed\n"); done = true; return; }
        for (int y = 0; y < H; ++y) memcpy(&rgb.px[(size_t)y * W * 3], rf.data[0] + (size_t)y * rf.stride[0], (size_t)W * 3);
        for (size_t i = 0; i < g.px.size(); ++i) {
            const uint8_t* p = &rgb.px[i * 3];
            g.px[i] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
        }
        std::vector<double> se, ss, sm, st, sc;
        double ye = method_edge(g, a.roi, a.strip, a.edge_h, &se);
        double ys = method_step(g, a.roi, a.strip, &ss);
        double ym = method_mirror(g, a.roi, a.strip, a.mirror_k, &sm);
        tv.push(g);
        double yt = tv.ready() ? tv.estimate(a.strip, &st) : NAN;
        Line L = method_line(g, a.roi, a.edge_h);
        double yg = method_gauge(rgb, a.gx0, a.gx1, a.roi.y0, a.roi.y1);
        double yc = method_color(rgb, a.roi, a.strip, &sc);

        fprintf(csv, "%d,%.1f,%.1f,%.1f,%.1f,%.2f,%.5f,%d,%.1f,%.1f\n", fi, ye, ys, ym, yt, L.at(gauge_cx), L.a, L.inliers, yg, yc);
        hist["A edge    "].push_back(ye);
        hist["B step    "].push_back(ys);
        hist["C mirror  "].push_back(ym);
        if (!std::isnan(yt)) hist["D temporal"].push_back(yt);
        hist["E line    "].push_back(L.at(gauge_cx));
        hist["F gauge   "].push_back(yg);
        hist["G color   "].push_back(yc);

        bool want_snap = !a.snap.empty() && !snapped && tv.ready();
        if (!a.overlay.empty() || want_snap) {
            Rgb o = rgb;
            rect(o, a.roi, {255, 255, 255});
            auto sx = strips(a.roi, a.strip);
            for (size_t i = 0; i < sx.size(); ++i) {
                // 短冊を 5 等分し、A B C D G の短冊ごとの値を順に短く描く
                int x0 = sx[i].first, x1 = sx[i].second, q = (x1 - x0) / 5;
                hline(o, x0, x0 + q, se[i], C_EDGE);
                hline(o, x0 + q, x0 + 2 * q, ss[i], C_STEP);
                hline(o, x0 + 2 * q, x0 + 3 * q, sm[i], C_MIRROR);
                if (i < st.size()) hline(o, x0 + 3 * q, x0 + 4 * q, st[i], C_TEMP);
                hline(o, x0 + 4 * q, x1, sc[i], C_COLOR);
            }
            seg(o, 0, W, L, C_LINE);
            hline(o, a.gx0 - 15, a.gx1 + 15, yg, C_GAUGE);
            hline(o, a.gx0 - 15, a.gx1 + 15, yg + 1, C_GAUGE);
            Color cs[] = {C_EDGE, C_STEP, C_MIRROR, C_TEMP, C_LINE, C_GAUGE, C_COLOR};
            for (int k = 0; k < 7; ++k) legend(o, 20 + k * 40, 640, cs[k]);
            if (!a.overlay.empty()) {
                mk::VideoFrame of;
                of.width = W; of.height = H; of.format = mk::PixelFormat::RGB24;
                of.buffer = o.px; of.data[0] = of.buffer.data(); of.stride[0] = W * 3;
                of.pts_us = (int64_t)(used * 1e6 * a.every / fps);
                enc_out.clear();
                if (!enc.encode(of, enc_out)) fprintf(stderr, "encode: %s\n", enc.error().c_str());
                for (auto& p : enc_out) writer.write(p);
            }
            if (want_snap) { write_bmp(a.snap, o); snapped = true; }
        }
        ++used;
        if (a.max_frames && used >= a.max_frames) done = true;
    };

    mk::Packet pkt;
    std::vector<mk::VideoFrame> frames;
    while (!done && reader.read(pkt)) {
        if (pkt.type != mk::MediaType::Video) continue;
        frames.clear();
        if (!dec.decode(pkt, frames)) { fprintf(stderr, "decode: %s\n", dec.error().c_str()); break; }
        for (auto& f : frames) handle(f);
    }
    if (!done) { frames.clear(); dec.flush(frames); for (auto& f : frames) handle(f); }
    if (!a.overlay.empty()) {
        enc_out.clear(); enc.flush(enc_out);
        for (auto& p : enc_out) writer.write(p);
        writer.close();
    }
    fclose(csv);

    printf("%s: %dx%d %.2f fps, frames analysed: %d (every %d)\n", a.in.c_str(), info.width, info.height, fps, used, a.every);
    printf("method       median     p10     p90    (y, px; smaller = higher water)\n");
    for (auto& [k, v] : hist) {
        if (v.empty()) continue;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        auto q = [&](double f) { return s[(size_t)std::min<double>((double)s.size() - 1, f * ((double)s.size() - 1))]; };
        printf("%s  %7.1f %7.1f %7.1f\n", k.c_str(), median(v), q(0.1), q(0.9));
    }
    return 0;
}
