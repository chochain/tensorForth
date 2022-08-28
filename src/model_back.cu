/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model Backward Propagation implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
__GPU__ Model&
Model::backprop(Tensor &tgt) {
    printf("here\n");
    Tensor &nx = (*this)[numel - 1];
    if (!nx.is_same_shape(tgt)) {
        ERROR("Model#backprop target dim?\n");
        return *this;
    }
    ///
    /// cascade execution layer by layer backward
    ///
    nx -= tgt;
    for (U16 i = numel - 2; i > 0; i--) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        printf("%2d> %s [%d,%d,%d] p=%d <=",
               i, d_nname(in.grad_fn), in.H(), in.W(), in.C(), in.parm); 
        _bstep(in, out);
        printf("\n");
    }
    return *this;
}
/// ========================================================================
/// private methods 
///
#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14 */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12 */

__GPU__ void
Model::_bstep(Tensor &in, Tensor &out) {
    DU   *da = in.data, *dc = out.data;              ///< input, output data
    int  H = out.H(), W = out.W(), C = out.C();      ///< output HWC
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, C), grd(        ///< GPU warp size setup
        (W + blk.x - 1) / blk.x,
        (H + blk.y - 1) / blk.y
    );
    auto dump = [](DU *v, int H, int W, int C) {
        for (int k = 0; k < C; k++) {
            printf("\nC=%d ---", k);
            for (int i = 0; i < H; i++) {
                printf("\n");
                for (int j = 0; j < W; j++) {
                    DU x = v[k + (j + i * W) * C];
                    printf(x < DU0 ? "%.2f" : " %.2f", x);
                }
            }
        }
        printf("\n");
    };
    ///
    /// layer function dispatcher
    ///
    printf(" out'[%d,%d,%d]", H, W, C);
    t4_layer fn = in.grad_fn;                 ///< layer function
    switch(fn) {
    case L_CONV:   {
        Tensor &f = *in.grad[0];              ///< filter tensor
        Tensor &b = *in.grad[1];              ///< bias tensor
        printf(" f[%d][%d,%d,%d,%d], b[%d]",
               f.parm, f.N(), f.H(), f.W(), f.C(), b.numel);
    } break;
    case L_LINEAR: {                          ///< out = w @ in + b
        Tensor &dw = *in.grad[2];             ///< weight tensor
        Tensor &db = *in.grad[3];             ///< bias tensor
        int M = dw.H(), N = dw.W();           ///< fully connected dimensions
        printf(" out'[%d] @ dw[%d,%d].t", out.numel, M, N);
        if (db.numel == out.numel) {
            db += out;
        }
        else ERROR("db, out' dim?\n");
        dump(db.data, 1, db.numel, 1);
    } break;
    case L_FLATTEN: break;
    case L_RELU:    break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: break;
    case L_DROPOUT: break;
    }
    cudaDeviceSynchronize();
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
