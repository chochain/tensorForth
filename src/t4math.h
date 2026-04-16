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
#define TM       4                    /** register tile W dim */
#define TN       4                    /** register tile H dim */
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
///@}
///@name Numeric conversion
///@{
__GPU__ inline float d__stride_sum(float *src, long numel, long tid);      /// stride sum per thread
__GPU__ inline float d__stride_var(float *src, float avg, long numel, long tid);
__GPU__ inline float d__warp_sum(float v);                                 /// reduce sum up a warp
__GPU__ inline float d__rollup_sum(float *smem);
///@}
///@name Tensor ops (kernel mode)
///@{
__KERN__ void k_sum(float* __restrict__ src, float* __restrict__ sum, long numel);
__KERN__ void k_nvar(float *src, float *avg, float var, long numel);       /// n * variance
__KERN__ void k_batchsum(float *src, float *sum, long numel);
__KERN__ void k_batchnvar(float *src, float *avg, float *var, long numel);
__KERN__ void k_copy(float *src, float *dst, long n);                      ///< Note: (src, dst)
__KERN__ void k_transpose(float *src, float *dst, int h, int w);           ///< Note: (src, dst), TODO: CDP
__KERN__ void k_identity(float *t, int h, int w);
__KERN__ void k_math(math_op op, float *dst, float v, long n);             ///< tensor math ops
__KERN__ void k_ts_op(math_op op, float *A, float v, float *O, long n);    ///< tensor-scalar ops
__KERN__ void k_tt_op(math_op op, float *A, float *B, float *O, long n);   ///< tensor-tensor ops
__KERN__ void k_bce(float *O, float *T, long n);
///@}    
///@name Tensor debug ops (kernel mode)
///@{
__KERN__ void k_nan_inf(float *src, int *n, long numel);
__KERN__ void k_dummy();
///@}
///@}
///@name BLAS ops
///@{    
__KERN__ void k_matmul(
    float *A, float *B, float *O,                                ///< O[M*N*C] = A[M*K*C] @ B[K*N*C]
    bool tA, bool tB, bool inc, int K, int M, int N);
__KERN__ void k_gemm(                                   ///< O[M*N*C] = a * A[M*K*C] @ B[K*N*C] + b * O[M*N*C]
    float *A, float *B, float *O,                            
    float alpha, float beta, bool tA, bool tB, int K, int M, int N);  
__KERN__ void k_gemm_claude(
    const float * __restrict__ A, const float * __restrict__ B, float *O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N);
__KERN__ void k_gemm_tile_gemini(
    float *__restrict__ A, float *__restrict__ B, float *O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N);
__KERN__ void k_gemm_tile_claude(
    float * __restrict__ A, float * __restrict__ B, float *O,
    float alpha, float beta,  bool tA, bool tB, int K, int M, int N);
    
} // namespace t4
#endif // __T4MATH_H_
    



