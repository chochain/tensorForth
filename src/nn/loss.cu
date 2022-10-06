/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
__GPU__ Tensor&
Model::onehot() {
    auto show = [](DU *h, int n, int sz) {
        printf("onehot[%d]=", n);
        for (int i = 0; i < sz; i++) {
            printf("%2.0f", h[i]);
        }
        printf("\n");
    };
    Tensor &out = (*this)[-1];                         ///< model output
    int    N    = out.N(), hwc = out.HWC();            ///< sample size
    Tensor &hot = _t4(N, hwc).fill(DU0);               ///< one-hot vector
    if (!_dset) {
        ERROR("Model#loss dataset not set yet?\n");
        return hot;
    }
    for (int n = 0; n < N; n++) {                      /// * loop through batch
        DU *h = hot.slice(n);                          ///< 
        U32 i = INT(_dset->label[n]);
        h[i < hwc ? i : 0] = DU1;
        show(h, n, hwc);
    }
    return hot;
}

__GPU__ DU
Model::loss(t4_loss op) {
    return loss(op, *_hot);                     /// * use default one-hot vector
}
__GPU__ DU
Model::loss(t4_loss op, Tensor &hot) {          ///< loss against one-hot
    Tensor &out = (*this)[-1];                  ///< model output
    if (!out.is_same_shape(hot)) {              /// * check dimensions
        ERROR("Model#loss hot dim != out dim\n");
        return;
    }
    Tensor &tmp = _mmu->copy(out);              ///< non-destructive
    DU err = _loss(op, tmp, hot);               /// * calculate loss
    _mmu->free(tmp);                            /// * free memory

    return err;
}
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
            printf("\nn=%d\n", n);
            _dump(d, H, W, C);
        }
        if (sz > 36) _view(d, H, W, C, scale);
    }
}

__GPU__ void
Model::_view(DU *v, int H, int W, int C, DU scale) {
//  static const char *map = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";   // 69 shades
    static const char *map = " .:-=+*#%@X";
    const int sz = H * W, sq = (int)sqrt(sz);
    const int sh = (sz/sq) + ((sz - sq*sq) > 0 ? 1 : 0);
    const int h  = W > 1 ? H : (sz < 36 ? 1 : sh);
    const int w  = W > 1 ? W : (sz < 36 ? H : sq);
    for (int i = 0; i < h; i++) {
        printf("\n");
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < w; j++) {
                int n = j + i * w;
                if (n < sz) { 
                    DU x0 = (v[k + (j>0 ? n - 1 : n) * C]) * scale;
                    DU x1 = (x0 + v[k + n * C] * scale) * 0.5;
                    char c0 = map[x0 < 10.0f ? (x0 < DU0 ? 10 : (int)x0) : 9];
                    char c1 = map[x1 < 10.0f ? (x1 < DU0 ? 10 : (int)x1) : 9];
                    printf("%c%c", c0, c1);                           // double width
                }
                else printf("  ");
            }
            printf("|");
        }
    }
    printf("\n");
}
__GPU__ void
Model::_dump(DU *v, int H, int W, int C) {
    const int sz = H * W, sq = (int)sqrt(sz);
    const int sh = (sz/sq) + ((sz - sq*sq) > 0 ? 1 : 0);
    const int h  = W > 1 ? H : (sz < 36 ? 1 : sh);
    const int w  = W > 1 ? W : (sz < 36 ? H : sq);
    for (int i = 0; i < h; i++) {
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < w; j++) {
                int n = j + i * w;
                if (n < sz) printf("%5.2f", v[k + n * C]);
                else        printf(" ....");
            }
            printf("|");
        }
        printf("\n");
    }
}
__GPU__ void
Model::_dump_dbdf(Tensor &db, Tensor &df) {
    const int fsz = df.N() * df.H() * df.W() * df.C();
    const int C5  = df.parm, C0 = db.H();
    DU sum = DU0;
    printf("\n\tdb=");
    for (int c0 = 0; c0 < C0; c0++) {
        printf("%6.3f ", db.data[c0]);
        sum += db.data[c0];
    }
    printf("Σ=%6.3f", sum);
    DU *p = df.data;
    for (int c5 = 0; c5 < C5; c5++) {
        printf("\n\tdf[%d]=", c5);
        sum = DU0;
        for (int i=0; i<fsz; i++, p++) {
            sum += *p;
            printf("%6.3f", *p);
        }
        printf(" Σ=%6.3f", sum);
    }
}
__GPU__ DU
Model::_loss(t4_loss op, Tensor &out, Tensor &hot) {
    DU err = DU0;                    ///> result loss value
    switch (op) {
    case LOSS_MSE:                   /// * mean squared error, input from linear
        out -= hot;
        err = 0.5 * NORM(out.numel, out.data) / out.numel;
        break;
    case LOSS_CE:                    /// * cross_entropy, input from softmax
        out.map(O_LOG);              /// * out = log-softmax now
        /* no break */
    case LOSS_NLL:                   /// * negative log likelihood, input from log-softmax
        out *= hot;                  /// * multiply probability of two systems
        err = -out.sum() / out.N();  /// * negative average per sample
        break;
    default: ERROR("Model#loss op=%d not supported\n", op);
    }
    SCALAR(err);
    return err;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
