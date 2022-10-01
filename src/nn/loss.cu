/** -*- c++ -*-
 * @file
 * @brief Model class - loss and trace functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
__GPU__ DU
Model::loss(t4_loss op, Tensor &tgt) {
    Tensor &nx = (Tensor&)_mmu->du2obj(data[numel - 1]);  ///< model output 
    if (!nx.is_same_shape(tgt)) { ERROR("Model::loss dim?\n"); return; }

    Tensor &out = _mmu->copy(nx);                  ///> tmp, hardcopy
    DU     err  = DU0;                             ///> result loss value
    switch (op) {
    case LOSS_NLL: break;
    case LOSS_MSE: {
        out -= tgt;
        err = 0.5 * NORM(out.numel, out.data) / out.numel;
    } break;
    case LOSS_CE:  {
        out.map(O_LOG) *= tgt;
        err = -out.avg();
    } break;
    default: ERROR("Model#loss op=%d not supported\n", op);
    }
    _mmu->free(out);                              /// * release the tmp
    SCALAR(err);
    return err;
}
///
/// debug dumps
///
__GPU__ void
Model::view(DU *v, int H, int W, int C) {
//  static const char *map = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";   // 69 shades
    static const char *map = " .:-=+*#%@";
    static const DU   X    = 10.0f;
    for (int k = 0; k < C; k++) {
        printf("\nC=%d ---", k);
        for (int i = 0; i < H; i++) {
            printf("\n");
            for (int j = 0; j < W; j++) {
                DU   x0 = (v[k + ((j>0 ? j-1 : j) + i * W) * C]) * X;
                DU   x1 = (x0 + v[k + (j + i * W) * C] * X) * 0.5;
                char c0 = map[x0 < X ? (x0 >= DU0 ? (int)x0 : 0) : 9];
                char c1 = map[x1 < X ? (x1 >= DU0 ? (int)x1 : 0) : 9];
                printf("%c%c", c0, c1);                // double width
            }
        }
    }
    printf("\n");
}
__GPU__ void
Model::dump(DU *v, int H, int W, int C) {
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
Model::dump_dbdf(DU *df, DU *db, int C0, int C1, int fsz) {
    DU sum = DU0;
    printf("\n\tdb=");
    for (int c0 = 0; c0 < C0; c0++) {
        printf("%6.3f ", db[c0]);
        sum += db[c0];
    }
    printf("Σ=%6.3f", sum);
    for (int c1 = 0; c1 < C1; c1++) {
        printf("\n\tdf[%d]=", c1);
        sum = DU0;
        for (int i=0; i<fsz; i++, df++) {
            sum += *df;
            printf("%6.3f", *df);
        }
        printf(" Σ=%6.3f", sum);
    }
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
