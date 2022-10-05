/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
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
    const int H  = t.H(), W = t.W(), C = t.C();
    const int sq = (int)sqrt(0.5f + H);
    
    for (int n = 0; n < t.N(); n++) {
        DU *d = t.slice(n);
        if (W > 1) _view(d, H, W, C);
        else {
            if (sq > 6) _view(d, sq, sq, C, scale);
            _dump(d, W, H, C);
        }
    }
}

__GPU__ void
Model::_view(DU *v, int H, int W, int C, DU scale) {
//  static const char *map = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";   // 69 shades
    static const char *map = " .:-=+*#%@X";
    for (int i = 0; i < H; i++) {
        printf("\n");
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < W; j++) {
                DU x0 = (v[k + ((j>0 ? j-1 : j) + i * W) * C]) * scale;
                DU x1 = (x0 + v[k + (j + i * W) * C] * scale) * 0.5;
                char c0 = map[x0 < 10.0f ? (x0 < DU0 ? 10 : (int)x0) : 9];
                char c1 = map[x1 < 10.0f ? (x1 < DU0 ? 10 : (int)x1) : 9];
                printf("%c%c", c0, c1);                // double width
            }
            printf("|");
        }
    }
    printf("\n");
}
__GPU__ void
Model::_dump(DU *v, int H, int W, int C) {
    for (int k = 0; k < C; k++) {
        printf("\nC=%d ---", k);
        for (int i = 0; i < H; i++) {
            printf("\n");
            for (int j = 0; j < W; j++) {
                DU x = v[k + (j + i * W) * C];
                printf("%5.2f", x);
            }
        }
    }
    printf("\n");
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
