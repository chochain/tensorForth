/** -*- c++ -*-
 * @file
 * @brief Neural Network Math module
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __NN_NMATHL_H
#define __NN_NMATHL_H
#pragma once
#include <float.h>
#include "ten4_config.h"
///
/// General Notations
///   xx1  : 1 stands for input,  i.g. i1 - x dimension index for input
///   xx0  : 0 stands for output, i.g. C0 - number of channels for output
///   n, N : index in mini-batch
///   c, C : index among channels
///   i, W : index in x dimension
///   j, H : index in y dimension
///   k    : general counter
///   z, HW: page index
///
/// TODO: stride, dilation, [C1]NCHW filter,
///       [see](https://github.com/vdumoulin/conv_arithmetic)
/// 
#if (T4_DO_OBJ && T4_DO_NN)

namespace t4::nn {
typedef float     DU;
typedef uint32_t  U32;

#define SELU_L  1.0507                     /** Selu lambda */
#define SELU_LA 1.7581                     /** Selu alpha  */
///
/// forward
///
template<int TS, int R>                     ///> tile size, kernel radius
__KERN__ void k_conv2d(
    DU *I, DU *F, DU *B, DU *O,             ///> input I[HxW], F[KxK] kernel, B[C] bias, output O[HxW]
    U32 H, U32 W, U32 C1, U32 C0);          ///< (H0==H1, W0==W1), input/output channels
template<U32 KS>
__KERN__ void k_pool(t4_layer op, DU *I, DU *O, U32 H, U32 W);
__KERN__ void k_bias(DU *B, DU *O, U32 N, U32 E0);
__KERN__ void k_activate(
    t4_layer op, DU *I, DU *F, DU *O,       ///< func, input, filter, output tensors
    DU alpha, U64 numel);                   ///< number of tensor elements
__KERN__ void k_softmax_small(DU *I, DU *O, U32 C);  ///< one block per sample, C ≤ 256
__KERN__ void k_softmax(DU *I, DU *O, U32 C);
__KERN__ void k_batchnorm(
    DU *I, DU *O,  DU *X,                   ///< input, filter, output tensors
    DU *avg, DU *rvar,                      ///< mean, 1.0/(stdvar + e)
    DU *w, DU *b, U64 HW);                  ///< gamma, beta, H0=H1, W0==W1 (C0==C1)
///
/// backprop
///
template<U32 TS, U32 R>
__KERN__ void k_dconv2d(
    DU *I, DU *DX, DU *F, DU *DF, DU *DB, DU *O,
    U32 H, U32 W, U32 C0, U32 C1, bool train);
template<U32 KS>
__KERN__ void k_dpool(t4_layer op, DU *I, DU *O, U32 H, U32 W);
__KERN__ void k_dlinear_db(DU *O, DU *DB, U32 N, U32 E0);
__KERN__ void k_dbatchnorm_1(
    DU *I, DU *O, DU *X,                    ///< input, output, x_hat tensors
    DU *sum, DU *g_var, U64 HW);            ///< sum(x_hat), gamma/(stdvar+e)
__KERN__ void k_dbatchnorm_2(
    DU *I, DU *X, DU *sum, U64 HW);         ///< input, x_hat, H0=H1, W0==W1 (C0==C1)
///
/// * gradient
///
__KERN__ void k_sgd(
    DU *G, DU *DG, DU *M,                    ///< w, dw, and momemtum tensors
    U32 N, DU lr, DU b, U64 numel);          ///< batch size, learn rate, beta(momemtum)
__KERN__ void k_adam(
    DU *G, DU *DG, DU *M, DU *V,             ///< w, dw, and momemtum tensors
    U32 N, DU lrc, DU b1, DU b2, U64 numel); ///< batch size,corrected learn rate, beta(momemtum)

} // namespace t4::nn

#endif  // (T4_DO_OBJ && T4_DO_NN)
#endif // __NN_MODEL_H
//==========================================================================
