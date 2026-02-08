/** -*- c++ -*-
 * @file
 * @brief Model class - debug/trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "dataset.h"

__GPU__ int
Model::_check_nan(Tensor &t) {
    static int cnt; cnt = 0;
    FORK1(k_nan_inf, t.numel, t.data, &cnt);
    GPU_SYNC();
    return cnt;
}

__GPU__ void
Model::_dump_f(const char *fn, Tensor &f) {
    DU sum = DU0;
    for (U32 n = 0; n < f.N(); n++) {
        DU *v = f.slice(n), fsum = DU0;
        INFO("\n\t%s[%d]=", fn, n);
        for (U32 i = 0; i < f.HWC(); i++) {
            sum  += *v;
            fsum += *v;
            INFO("%5.2f ", *v++);
        }
        INFO("Σ=%5.2f", fsum);
    }
    INFO("\n\tΣΣ=%6.3f", sum);
}

__GPU__ void
Model::_dump_b(const char *bn, Tensor &b) {
    DU sum = DU0;
    INFO("\n\t%s=", bn);
    DU *v = b.data;
    for (U32 i = 0; i < b.H(); i++) {
        INFO("%6.3f ", *v);
        sum += *v++;
    }
    INFO(" Σ=%6.3f", sum);
}

__GPU__ void
Model::_dump_w(const char *wn, Tensor &w, bool full) {
    const U32 H = w.H(), W = w.W();
    DU hsum = DU0, *p = w.data;
    if (full) INFO("\n%s[%d,%d]=", wn, H, W);
    else      INFO("\n\tdwΣ=");
    for (U32 i = 0; i < H; i++) {
        if (full) INFO("\n\t%s[%d]=", wn, i);
        DU sum = DU0;
        for (U32 j = 0; j < W; j++, p++) {
            sum  += *p;
            hsum += *p;
            if (full) INFO("%6.3f", *p);
        }
        if (full) INFO(" Σ=%6.3f", sum);
        else      INFO("%6.3f ", sum);
    }
    if (H > 1) INFO("%sΣΣ=%6.3f", full ? "\n\t" : " ", hsum);
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
