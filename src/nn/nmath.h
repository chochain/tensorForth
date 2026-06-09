/** -*- c++ -*-
 * @file
 * @brief Neural Network Math module
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __NN_NMATH_H
#define __NN_NMATH_H
#pragma once
#include "ten4_types.h"
#include "t4math.h"
#include "ntypes.h"
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
namespace t4::nn {

#if (T4_DO_OBJ && T4_DO_NN)

#define SELU_L  1.0507                     /** Selu lambda */
#define SELU_LA 1.7581                     /** Selu alpha  */
///============================================================================
/// forward
///============================================================================
/// template<int TS, int R> void k_conv2d   ///> include at the end of this file
/// template<int KS>        void k_pool
///
__KERN__ void k_bias(
    DP_R B, DP_W O, int N, int E0);
__KERN__ void k_activate(
    t4_layer op,                            ///< function to call 
    DP_R I, DP_W O, DP_W F,                 ///< input, output, filter tensors
    DU alpha, long numel);                  ///< number of tensor elements
__KERN__ void k_softmax_small(              ///< one block per sample, C ≤ 256
    DP_R I, DP_W O, int C);
__KERN__ void k_softmax(
    DP_R I, DP_W O, int C);
__KERN__ void k_batchnorm_stat(
    DP_R src, DP_W avg, DP_W var, long HW); ///< input  [N, HW, C] (NHWC)
__KERN__ void k_batchnorm_calc(             ///< copy avg, var per C
    DP_X avg, DP_X var, long NHW);
__KERN__ void k_batchnorm(
    DP_R I, DP_W O, DP_X X,                 ///< input, filter, output tensors
    DP_R avg, DP_R rvar,                    ///< mean, 1.0/(stdvar + e)
    DP_R w, DP_R b,                         ///< gamma, beta
    long HW);                               ///< H0=H1, W0==W1 (C0==C1)
///============================================================================
/// backprop
///============================================================================
/// template<int TS, int R> void k_dconv2d  /// included at the end of 
/// template<int KS>        void k_dpool
///
__KERN__ void k_dlinear_db(
    DP_R O, DP_W DB, int N, int E0);
__KERN__ void k_dbatchnorm_1(
    DP_W I, DP_X O, DP_R X,                 ///< input, output, x_hat tensors
    DP_R sum, DP_R g_var, long HW);         ///< sum(x_hat), gamma/(stdvar+e)
__KERN__ void k_dbatchnorm_2(
    DP_W I, DP_R X, DP_R sum, long HW);     ///< input, x_hat, H0=H1, W0==W1 (C0==C1)
///
/// * gradient
///
__KERN__ void k_sgd(
    DP_X G, DP_X DG, DP_X M,                 ///< w, dw, and momemtum tensors
    int N,                                   ///< batch size
    DU lr, DU b,                             ///< learn rate, beta(momemtum)
    long numel);
__KERN__ void k_adam(
    DP_X G, DP_X DG, DP_X M, DP_X V,         ///< w, dw, and momemtum tensors
    int N,                                   ///< batch size
    DU lrc, DU b1, DU b2,                    ///< corrected learn rate, beta(momemtum)
    long numel);

#include "nmath.tcu"                         ///< templates

#endif  // (T4_DO_OBJ && T4_DO_NN)
} // namespace t4::nn

#endif // __NN_NMATH_H
//==========================================================================
