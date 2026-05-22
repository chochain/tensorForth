/** -*- c++ -*-
 * @file
 * @brief Tensorboard class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mu/tensor.h"
#include "nn/model.h"
#include "summary.h"

namespace t4::tb {

#if T4_DO_TB

__HOST__ void
Summary::init(const char *run_id) {
    if (run_id && strcmp(run_id, _run_id) != 0) {
        teardown();
        _run_id = run_id;
    }
    std::string rundir  = std::string(_root) + "/" + _run_id;
    std::string logname = _logname(rundir);
    mkdir(rundir.c_str(), 0755);                  /// * create Event/Run subdir
    EventWriter::setup(logname.c_str());
}

__HOST__ void
Summary::image(const char *tag, Tensor &t) {
    const U32 W = t.W(), H = t.H(), C = t.C();
    const DU  mean = t.avg(), scale = (t.std() - 0.5f) * 128.0f;  /// 95%
    U8V px(W * H * 3);
    DU  hx[W * H * C];
    for (int n = 0; n < t.N(); n++) {
        DU *d = t.slice(n), *h = hx;
        D2H(h, d, sizeof(DU) * W * H * C);
        for (int y = 0; y < H; y++) {
            U8 *p = &px[(y * W) * 3];
            for (int x = 0; x < W; x++, h++) {
                DU vx = (*h - mean) * scale + 128.5f;
                U8 v  = (U8)MIN(255, MAX(vx, 0));
                *p++ = v;
                *p++ = v;
                *p++ = v;
            }
        }
        add_image(tag, W, H, px, _step);
    }
}

__HOST__ void
Summary::tile(const char *tag, Tensor &t, int n_per_row) {
    const U32  N     = t.N(), H  = t.H(), W = t.W(), C = t.C();
    const int  WT    = n_per_row * W;
    const int  HT    = (N + n_per_row - 1) / n_per_row;
    const DU   mean  = t.avg();
    const DU   scale = (64.0f / t.std());        /// 2 std = 95%

    auto tile = [&](U8V &px, DU *v, int idx) {
        int ht = idx / n_per_row, wt = idx % n_per_row;
        U8 *p = &px[(ht * H * WT + wt * W) * 3];
        for (int y = 0; y < H; y++) {
            for (int x = 0, c = 0; x < W; x++, c=0) {
                while (c < 3) {                  /// RGB
                    DU vx = (*v - mean) * scale + 128.0f;
                    *p++ = (x==0 && y==0)
                        ? 128 : static_cast<U8>(MIN(255, MAX(vx, 0)));
                    if (c++ < C) v++;            /// advance if more than 1 channel
                }
            }
            p += (WT - W) * 3;                   /// skip to next row in tile
        }
    };

    U8V px((HT * H) * WT * 3);                  ///< zero-init, so unfilled are black
    DU  h[t.numel];                             ///< host block (watch out, heap space)
    D2H(h, t.data, sizeof(DU) * t.numel);       /// * copy to host
    for (int n = 0; n < N; n++) {
        DU *hx = &h[n * H * W * C];
        tile(px, hx, n);                         /// * fill tile by tile
    }
    add_image(tag, WT, H * HT, px, _step);
}

__HOST__ void
Summary::histo(const char *tag, Tensor &t, int n_bucket) {
    add_histo(tag, t.data, t.numel, _step, n_bucket);
}

__HOST__ void
Summary::graph(const char *tag, Model &m) {
    init_graph();
    for (int i = 0; i < m.numel; i++) {
        Tensor &in = m[i], &out = m[i + 1];
    }
    add_graph();
}

#endif // T4_DO_TB

} // namespace t4::tb

