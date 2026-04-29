/**
 * @file
 * @brief Math/Blas utility functions interface
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __T4MATH_H_
#define __T4MATH_H_
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "ten4_config.h"

#define __HOST__     __host__
#define __KERN__     __global__
#define __GPU__      __device__

namespace t4 {

///@name cross platform floating-point ALU support (see nvcc -use_fast_math flag)
///@{
typedef enum {
    /// 1-operand arithmetic ops
    ABS = 0,
    NEG,
    EXP,
    LN,
    LOG,
    TANH,
    RELU,
    SIGM,
    SQRT,
    RCP,
    SAT,
    IDEN,
    /// 1-operand + a constant
    FILL,
    GFILL,
    SCALE,
    POW,
    /// 2-operand ops
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    MAX,
    MIN,
    MUL2,
    MOD2
} math_op;

#define MATH_OP "abs","neg","exp","ln","log","tanh","relu","sigmoid","sqrt","rcp","sat","iden","fill","gfill","scale","pow","+","-","*","/","mod","max","min","mul2","mod2"

#if 0 // GPU mode __CUDA_ARCH__

#define ABS(d)       (fabsf(d))                 /**< absolute value         */
#define NEG(d)       (-d)                       /**< negate                 */
#define EXP(d)       (__expf(d))                /**< exponential(float)     */
#define LN(d)        (__logf(d))                /**< natural logrithm       */
#define LOG(d)       (__log10f(d))              /**< log10                  */
#define TANH(d)      (atanhf(d))                /**< tanh(float)            */
#define RELU(d)      (MAX(0.0, d))              /**< relu(float)            */
#define SIGMOID(d)   (RCP(1.0+EXP(-(d))))       /**< sigmoid(float)         */
#define SQRT(d)      (__fsqrt_rn(d))            /**< square root            */
#define RCP(x)       (__frcp_rn(x))             /**< reciprocol 1/x         */
#define SAT(d)       (__saturatef(d))           /**< clamp into [0.0..1.0]  */
#define POW(d,e)     (__powf(d,e))              /**< power d^(e)            */
#define ADD(x,y)     (__fadd_rn(x,y))           /**< addition               */
#define SUB(x,y)     (__fsub_rn(x,y))           /**< subtraction            */
#define MUL(x,y)     (__fmul_rn(x,y))           /**< multiplication         */
#define DIV(x,y)     (__fdiv_rn(x,y))           /**< division               */
#define MOD(t,n)     (fmodf(t,n))               /**< fmod two floats        */
#define MAX(x,y)     (fmax(x,y))                /**< maximum of the two     */
#define MIN(x,y)     (fmin(x,y))                /**< minimum of the two     */
#define MUL2(x2,y2)  (__dmul_rn(x2,y2))         /**< double precision mul   */
#define MOD2(x2,y2)  (fmod(x2,y2))              /**< double precision mod   */

#else // (HOST mode) !__CUDA_ARCH__

#include <cmath>
#define ABS(d)       (fabs(d))                  /**< absolute value         */
#define NEG(d)       (-d)                       /**< negate                 */
#define EXP(d)       (expf(d))                  /**< exponential(float)     */
#define LN(d)        (logf(d))                  /**< natural logrithm       */
#define LOG(d)       (log10f(d))                /**< log10                  */
#define TANH(d)      (atanhf(d))                /**< tanh(float)            */
#define RELU(d)      (MAX(0.0f, (d)))           /**< relu(float)            */
#define SIGMOID(d)   (RCP(1.0f+EXP(-(d))))      /**< sigmoid(float)         */
#define SQRT(d)      (sqrtf(d))                 /**< square root            */
#define RCP(x)       (1.0f/(x))                 /**< reciprocol 1/x         */
#define SAT(d)       (MIN(1.0f,MAX(0.0f,(d))))  /**< clamp into [0.0..1.0]  */
#define POW(d,e)     (powf((d),(e)))            /**< power d^(e)            */
#define ADD(x,y)     ((x)+(y))                  /**< addition               */
#define SUB(x,y)     ((x)-(y))                  /**< subtraction            */
#define MUL(x,y)     ((x)*(y))                  /**< multiplication         */
#define DIV(x,y)     ((x)/(y))                  /**< division               */
#define MOD(t,n)     ((DU)fmodf((t),(n)))       /**< fmod two floats        */
#define MAX(x,y)     (fmaxf((x),(y)))           /**< maximum of the two     */
#define MIN(x,y)     (fminf((x),(y)))           /**< minimum of the two     */
#define MUL2(x2,y2)  ((DU2)(x2)*(y2))           /**< double precision mul   */
#define MOD2(x2,y2)  (fmod((DU2)(x2),(DU2)(y2)))/**< double precision mod   */

#endif // (GPU mode) __CUDA_ARCH__

typedef enum {
    T_DROP = 0,
    T_KEEP
} t4_drop_opt;
///@}
///@name GEMM Tiling parameters
///@{
#define TM      4                     /** register tile W dim */
#define TN      4                     /** register tile H dim */
#define BM      (T4_DIM_SZ * TN)      /** threads, tiled W dimension */
#define BN      (T4_DIM_SZ * TM)      /** threads, tiled H dimension */
#define BK      T4_DIM_SZ             /** [64,16] x [16,64] */

// ---------------------------------------------------------------------------
// FORK3T — grid over (ceil(W/BN), ceil(H/BM), C)
// ---------------------------------------------------------------------------
#define FORK3T(fn,h,w,c,...) {               \
    const dim3 _b(T4_DIM_SZ, T4_DIM_SZ, 1);  \
    const dim3 _g(((w) + BN - 1) / BN,       \
                  ((h) + BM - 1) / BM, c);   \
    fn<<<_g,_b>>>(__VA_ARGS__,h,w);          \
    GPU_CHK();                               \
}
#define F32_RP   const float * __restrict__
#define F32_WP         float * __restrict__
#define F32_XP         float *
///@}
///@name Tensor ops (kernel mode)
///@{
__KERN__ void k_sum(F32_RP src, F32_WP sum, long numel);
__KERN__ void k_nvar(F32_RP src, float avg, F32_WP var, long numel);       ///< n * variance
__KERN__ void k_max(F32_RP src, F32_WP rst, bool find_max, long numel);    ///< find_max=true max or false=min
__KERN__ void k_batchsum(F32_RP src, F32_WP sum, long numel);
__KERN__ void k_batchnvar(F32_RP src, F32_RP avg, F32_WP var, long numel);
__KERN__ void k_copy(F32_RP src, F32_WP dst, long n);                      ///< Note: (src, dst)
__KERN__ void k_transpose(F32_RP src, F32_WP dst, int h, int w);           ///< Note: (src, dst), TODO: CDP
__KERN__ void k_identity(F32_WP T, int h, int w);
__KERN__ void k_math(math_op op, F32_XP dst, float v, long n);             ///< tensor math ops
__KERN__ void k_ts_op(math_op op, F32_XP A, float v, F32_XP O, long n);    ///< tensor-scalar ops
__KERN__ void k_tt_op(math_op op, F32_RP A, F32_RP B, F32_WP O, long n);   ///< tensor-tensor ops
__KERN__ void k_bce(F32_RP T, F32_XP O, long n);
///@}    
///@name Tensor debug ops (kernel mode)
///@{
__KERN__ void k_nan_inf(F32_RP src, int *n, long numel);
__KERN__ void k_dummy();
///@}
///@}
///@name BLAS ops
///@{
__KERN__ void k_dot(
    F32_RP A,  F32_RP B, F32_XP O,          ///< O[N,C] = a * A[N,1,K,C] * B[N,1,K,C]
    float alpha, float beta, int K, int C);
__KERN__ void k_gemm(                       ///< O[N,H,W,C] = a * A[N,H,K,C] @ B[N,K,W,C] + b * O[N,H,W,C]
    F32_XP A, F32_XP B, F32_XP O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N);  
__KERN__ void k_gemm_claude(
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N);
__KERN__ void k_gemm_tile_claude(
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta,  bool tA, bool tB, int K, int M, int N);
__KERN__ void k_gemm_tile_claude_x2(
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta,  bool tA, bool tB, int K, int M, int N);
///@}
///@name Matrix inversion ops - Gauss-Jordan, LU
///@{
__KERN__ void k_find_pivot(const float *da, int *d_pivot, int z, int K);
__KERN__ void k_swap_rows(float *da, float *di, int u, int z, int K);
__KERN__ void k_diag(float *da, float *di, int z, int K);
__KERN__ void k_elim(float *da, float *di, int z, int K);
__KERN__ void k_lu_col(float *da, int z, int K);
__KERN__ void k_fsub(const float *lu, const int *d_piv, float *di, int K);
__KERN__ void k_bsub(const float *lu, float *di, int K);
__KERN__ void k_logdet(const float *lu, float *d_logdet, int *d_sign, int K);
///@}
} // namespace t4
#endif // __T4MATH_H_
    



