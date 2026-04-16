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

namespace t4 {
#define __HOST__     __host__
#define __KERN__     __global__
#define __GPU__      __device__

///@name Universal data types
///@{
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef float    DU;
typedef double   DU2;
///@}
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

#ifdef __cplusplus
extern "C" {
#endif
    
///@name Numeric conversion
///@{
__GPU__ inline float d__stride_sum(float *src, long numel, long tid);          /// stride sum per thread
__GPU__ inline float d__stride_var(float *src, float avg, long numel, long tid);
__GPU__ inline float d__warp_sum(float v);                                     /// reduce sum up a warp
__GPU__ inline float d__rollup_sum(float *smem);
///@}
///@name Tensor ops (kernel mode)
///@{
__KERN__ void k_sum(DU* __restrict__ src, DU* __restrict__ sum, U64 numel);
__KERN__ void k_nvar(DU *src, DU *avg, DU var, U64 numel);       /// n * variance
__KERN__ void k_batchsum(DU *src, DU *sum, U64 numel);
__KERN__ void k_batchnvar(DU *src, DU *avg, DU *var, U64 numel);
__KERN__ void k_copy(DU *src, DU *dst, U64 n);                   ///< Note: (src, dst)
__KERN__ void k_transpose(DU *src, DU *dst, int h, int w);       ///< Note: (src, dst), TODO: CDP
__KERN__ void k_identity(DU *t, int h, int w);
__KERN__ void k_math(math_op op, DU *dst, DU v, U64 n);          ///< tensor math ops
__KERN__ void k_ts_op(math_op op, DU *A, DU v, DU *O, U64 n);    ///< tensor-scalar ops
__KERN__ void k_tt_op(math_op op, DU *A, DU *B, DU *O, U64 n);   ///< tensor-tensor ops
__KERN__ void k_bce(DU *O, DU *T, U64 n);
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
    DU *A, DU *B, DU *O,                            ///< O[M*N*C] = A[M*K*C] @ B[K*N*C]
    U32 K, U32 M, U32 N, bool tA, bool tB, bool inc);
__KERN__ void k_gemm(                               ///< O[M*N*C] = a * A[M*K*C] @ B[K*N*C] + b * O[M*N*C]
    DU *A, DU *B, DU *O,                            
    U32 K, U32 M, U32 N, DU alpha, DU beta, bool tA, bool tB);  
__KERN__ void k_gemm_claude(
    const DU * __restrict__ A, const DU * __restrict__ B, DU *O,
    U32 K, U32 M, U32 N, DU alpha, DU beta, bool tA, bool tB);
__KERN__ void k_gemm_tile_gemini(
    DU *__restrict__ A, DU *__restrict__ B, DU *O,
    U32 K, U32 M, U32 N, DU alpha, DU beta,  bool tA, bool tB);
__KERN__ void k_gemm_tile_claude(
    DU * __restrict__ A, DU * __restrict__ B, DU *O,
    U32 K, U32 M, U32 N, DU alpha, DU beta,  bool tA, bool tB);
    
#ifdef __cplusplus
}
#endif

} // namespace t4
#endif // __T4MATH_H_
    



