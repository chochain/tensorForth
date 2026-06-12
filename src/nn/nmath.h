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
/// @name forward kernel functions
/// @note - template functions included at the end of this file
///   template<TS,K,S,P> void k_conv2d(...)
///   template<K>        void k_pool(...)
///============================================================================
/// @{
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
///@}
///============================================================================
/// @name backprop kernel functions
/// @note - template functions included at the end of this file
///   template<TS,K,S,P> void k_dconv2d(...)
///   template<K>        void k_dpool(...)
///============================================================================
///@{
__KERN__ void k_dlinear_db(
    DP_R O, DP_W DB, int N, int E0);
__KERN__ void k_batchnorm_1(                ///< reduce
    DP_R dout, DP_R xhat,                   ///< upstream gradient, saved x_hat
    DP_W sum_dout,                          ///< out: Σ dout        [N*C]
    DP_W sum_dout_xhat,                     ///< out: Σ dout*x_hat  [N*C]
    long HW);                               ///< H*W spatial elements
__KERN__ void k_batchnorm_2(
    DP_R W,                                 ///< gamma  [C]
    DP_W DW, DP_W DB,                       ///< d_gamma, d_beta accumulators [C]
    DP_W sum_dout,                          ///< in: Σ dout  [N*C]  → out: gvar*mean_dout
    DP_W sum_dout_xhat,                     ///< in: Σ dout*x̂ [N*C] → out: gvar*mean_dout_xhat
    DP_R var,                               ///< 1/sqrt(var+e)  [C]
    int  N, long NHW, bool train);          ///< batch size
__KERN__ void k_dbatchnorm(                 ///< final update
    DP_W DX,                                ///< output gradient tensor   [N,H,W,C]
    DP_R dout,                              ///< upstream gradient        [N,H,W,C]
    DP_R xhat,                              ///< saved x_hat              [N,H,W,C]
    DP_R s1,                                ///< gvar * mean(dout)        [N,C]
    DP_R s2,                                ///< gvar * mean(dout * x_hat)[N,C]
    long HW);                               ///< H*W
///@}
///============================================================================
/// @name gradient kernel functions
///============================================================================
///@{
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
///@}
#include "nmath.tcu"                         ///< templates (EOF include)

#endif  // (T4_DO_OBJ && T4_DO_NN)
} // namespace t4::nn

#endif // __NN_NMATH_H
//==========================================================================
