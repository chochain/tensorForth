/** -*- c++ -*-
 * @file
 * @brief Model class - debug/trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
///
/// debug dumps
///
__GPU__ void
Model::debug(Tensor &t, DU scale) {
    const int N  = t.N(), H = t.H(), W = t.W(), C = t.C();
    const int sz = H * W;
    
    for (int n = 0; n < N; n++) {
        DU *d = t.slice(n);
        if (sz < 100) {
            printf("\nn=%d", n);
            _dump(d, H, W, C);
        }
        if (sz > 36) _view(d, H, W, C, scale);
    }
}
///
/// private methods
///
__GPU__ DU
Model::_loss(t4_loss op, Tensor &out, Tensor &hot) {
    const int N = out.N();
    DU  err = DU0;                   ///> result loss value
    switch (op) {
    case LOSS_MSE:                   /// * mean squared error, input from linear
        out -= hot;
        err = 0.5 * NORM(out.numel, out.data) / N;
        break;
    case LOSS_CE:                    /// * cross_entropy, input from softmax
        out.map(O_LOG);
        /* no break */
    case LOSS_NLL:                   /// * negative log likelihood, input from log-softmax
        out *= hot;                  /// * hot_i * log(out_i)
        err = -out.sum() / N;        /// * negative average per sample
        break;
    default: ERROR("Model#loss op=%d not supported\n", op);
    }
    // debug(out);
    SCALAR(err);
    return err;
}
__GPU__ void
Model::_view(DU *v, int H, int W, int C, DU scale) {
//  static const char *map = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";   // 69 shades
    static const char *map = " .:-=+*#%@X";
    const int sz = H * W, sq = (int)sqrt(sz);
    const int sh = (sz/sq) + ((sz - sq*sq) > 0 ? 1 : 0);
    const int h  = W > 1 ? H : (sz < 36 ? 1 : sh);
    const int w  = W > 1 ? W : (sz < 36 ? H : sq);
    
    DU *csum = new DU[C];
    for (int k = 0; k < C; k++) csum[k] = DU0;
    for (int i = 0; i < h; i++) {
        printf("\n");
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < w; j++) {
                int n = j + i * w;
                if (n >= sz) { printf("  "); continue; }
                
                DU r0 = v[k + (j>0 ? n - 1 : n) * C];
                DU r1 = v[k + n * C];
                DU x0 = r0 * scale;
                DU x1 = (r0 + r1) * scale * 0.5;
                char c0 = map[x0 < 10.0f ? (x0 < DU0 ? 10 : (int)x0) : 9];
                char c1 = map[x1 < 10.0f ? (x1 < DU0 ? 10 : (int)x1) : 9];
                
                printf("%c%c", c0, c1);                           // double width
                csum[k] += r1;
            }
            printf("|");
        }
    }
    if (h > 1) {
        printf("\nΣΣ=");
        for (int k = 0; k < C; k++) printf("%5.2f ", csum[k]);
    }
    printf("\n");
    
    delete csum;
}
__GPU__ void
Model::_dump(DU *v, int H, int W, int C) {
    const int sz = H * W, sq = (int)sqrt(sz);
    const int sh = (sz/sq) + ((sz - sq*sq) > 0 ? 1 : 0);
    const int h  = W > 1 ? H : (sz < 36 ? 1 : sh);
    const int w  = W > 1 ? W : (sz < 36 ? H : sq);
    
    DU *csum = new DU[C];
    for (int k = 0; k < C; k++) csum[k] = DU0;
    for (int i = 0; i < h; i++) {
        printf("\n");
        DU sum = DU0;
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < w; j++) {
                int n = j + i * w;
                if (n >= sz) { printf(" ...."); continue; }
                
                DU  r = v[k + n * C];
                printf("%5.2f", r);
                sum += r;
                csum[k] += r;
            }
            printf("|");
        }
        printf("Σ=%6.3f", sum);
    }
    if (h > 1) {
        printf("\nΣΣ=");
        for (int k = 0; k < C; k++) printf("%5.2f ", csum[k]);
    }
    printf("\n");
    delete csum;
}
__GPU__ void
Model::_dump_dbdf(Tensor &db, Tensor &df) {
    _dump_db(db);
    
    DU sum = DU0;
    for (int n = 0; n < df.N(); n++) {
        DU *v = df.slice(n), fsum = DU0;
        printf("\n\tdf[%d]=", n);
        for (int i = 0; i < df.HWC(); i++) {
            sum  += *v;
            fsum += *v;
            printf("%5.2f ", *v++);
        }
        printf("Σ=%5.2f", fsum);
    }
    printf("\n\tΣΣ=%6.3f", sum);
}

__GPU__ void
Model::_dump_db(Tensor &db) {
    DU sum = DU0;
    printf("\n\tdb=");
    DU *v = db.data;
    for (int i = 0; i < db.H(); i++) {
        printf("%6.3f ", *v);
        sum += *v++;
    }
    printf(" Σ=%6.3f", sum);
}

__GPU__ void
Model::_dump_dw(Tensor &dw, bool full) {
    const int H = dw.H(), W = dw.W();
    DU hsum = DU0, *p = dw.data;
    if (!full) printf("\n\tdwΣ=");
    for (int i = 0; i < H; i++) {
        if (full) printf("\n\tdw[%d]=", i);
        DU sum = DU0;
        for (int j = 0; j < W; j++, p++) {
            sum  += *p;
            hsum += *p;
            if (full) printf("%6.2f", *p);
        }
        if (full) printf(" Σ=%5.2f", sum);
        else      printf("%5.2f ", sum);
    }
    if (H > 1) printf("%sΣΣ=%6.3f", full ? "\n\t" : " ", hsum);
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
