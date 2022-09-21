/**
 * @file
 * @brief tensorForth macros and internal type definitions
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TEN4_TYPES_H_
#define TEN4_SRC_TEN4_TYPES_H_
#include "ten4_config.h"

///
///@name Debug tracing options
///@{
#define INFO(...)           printf(__VA_ARGS__)
#if T4_VERBOSE
#define DEBUG(...)          printf(__VA_ARGS__)
#else  // T4_VERBOSE
#define DEBUG(...)
#endif // T4_VERBOSE
#if T4_MMU_DEBUG
#define WARN(...)           printf(__VA_ARGS__)
#define OPN(...)            static const char *opn[] = { __VA_ARGS__ }
#else  // T4_MMU_DEBUG
#define WARN(...)
#define OPN(...)
#endif // T4_MMU_DEBUG
#define ERROR(...)          printf(__VA_ARGS__)
#define NA(msg)             ({ ERROR("method not supported: %s\n", msg); })
///@}
///@name CUDA support macros
///@{
#if defined(__CUDACC__)
#include <cuda.h>
#include <cooperative_groups.h>
#define __GPU__             __device__
#define __HOST__            __host__
#define __BOTH__            __host__ __device__
#define __KERN__            __global__
#define __INLINE__          __forceinline__

#define MUTEX_LOCK(p)       while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)       atomicExch((int *)&p, 0)

#define ASSERT(X) \
    if (!(X)) ERROR("ASSERT tid %d: line %d in %s\n", threadIdx.x, __LINE__, __FILE__);
#define GPU_SYNC()          { cudaDeviceSynchronize(); }
#define GPU_CHK()           { \
    cudaDeviceSynchronize(); \
    cudaError_t code = cudaGetLastError(); \
    if (code != cudaSuccess) { \
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
        cudaDeviceReset(); \
    } \
}
#define MM_ALLOC(...)      { cudaMallocManaged(__VA_ARGS__); GPU_CHK(); }
#define MM_FREE(m)         cudaFree(m)

namespace cg = cooperative_groups;
#define K_RUN(...)          { cudaLaunchCooperativeKernel(__VA_ARGS__); GPU_CHK(); }
#else  // defined(__CUDACC__)
#define __GPU__
#define __HOST__
#define __KERN__
#define __INLINE__          inline
#define ASSERT(X)           assert(x)
#endif // defined(__CUDACC__)

#define H2D                 cudaMemcpyHostToDevice
#define D2H                 cudaMemcpyDeviceToHost
///@}
///@name Portable types (Rust alike)
///@{
typedef uint64_t    U64;                    ///< 64-bit unsigned integer
typedef uint32_t    U32;                    ///< 32-bit unsigned integer
typedef uint16_t    U16;                    ///< 16-bit unsigned integer
typedef uint8_t     U8;                     ///< 8-bit  unsigned integer
typedef uintptr_t   UFP;                    ///< function pointer type

typedef int64_t     I64;                    ///< 64-bit signed integer
typedef int32_t     I32;                    ///< 32-bit signed integer
typedef int16_t     I16;                    ///< 16-bit signed integer

typedef double      F64;                    ///< double precision float
typedef float       F32;                    ///< single precision float
///@}
///@name CUDA specific macros
///@{
#define TGRID(w,h,c,b)  ((w)+(b).x-1)/(b).x,((h)+(b).y-1)/(b).y,((c)+(b).z-1)/(b).z
///@}
//===============================================================================
/// tensorForth common data types
///
///@name Forth instruction and data types
///@{
#define DSIZE       5                       /**< most significan bit    */
typedef U16         IU;                     /**< instruction unit       */
typedef F32         DU;                     /**< data unit              */
typedef F64         DU2;                    /**< double preciesion data */
#define DU0         0.0                     /**< default data value 0   */
#define DU1         1.0                     /**< default data value 1   */
#define DU_EPS      1.0e-6                  /**< floating point epsilon */
///
/// cross platform floating-point math support
///
#define ZERO(d)     (ABS(d) < DU_EPS)       /**< zero check             */
#define BOOL(d)     (ZERO(d) ? DU0 : -DU1)  /**< default boolean        */
#define ABS(d)      (fabsf(d))              /**< absolute value         */
#define EXP(d)      (expf(d))               /**< exponential(float)     */
#define LOG(d)      (logf(d))               /**< natural logarithm      */
#define POW(d,e)    (powf(d,e))             /**< power d^(e)            */
#define SQRT(d)     (sqrtf(d))              /**< square root            */
#define TANH(d)     (tanhf(d))              /**< tanh(float)            */
#define MOD(t,n)    (fmodf(t, n))           /**< fmod two floats        */
#define DIV(x,y)    (fdividef(x,y))         /**< fast math devide       */
#define MAX(x,y)    (fmaxf(x,y))            /**< maximum of the two     */
#define MIN(x,y)    (fminf(x,y))            /**< minimum of the two     */
#define NORM(n,p)   (normf(n,p))            /**< normal of n floats     */
///
/// macros for object classification
///
#define T4_OBJ_FLAG 1                                /**< obj flag       */
#define DU2X(v)     (*(U32*)&(v))                    /**< to U32 ptr     */
#define SCALAR(v)   ((DU2X(v) &= ~T4_OBJ_FLAG), (v)) /**< clear obj flag */
#if     T4_ENABLE_OBJ
#define IS_OBJ(v)   (DU2X(v) & T4_OBJ_FLAG)          /**< if is an obj   */
#else  // T4_ENABLE_OBJ
#define IS_OBJ(v)   (0)
#endif // T4_ENABLE_OBJ
///@}
///
/// colon word compiler
/// Note:
///   * we separate dict and pmem space to make word uniform in size
///   * if they are combined then can behaves similar to classic Forth
///   * with an addition link field added.
///
enum {
    EXIT = 0, DONEXT, DOVAR, DOLIT, DOSTR, DOTSTR, BRAN, ZBRAN, DOES, TOR
} forth_opcode;

struct Managed {
    void *operator new(size_t sz) {
        void *ptr;
        cudaMallocManaged(&ptr, sz);
        GPU_SYNC();
        return ptr;
    }
    void operator delete(void *ptr) {
        GPU_SYNC();
        cudaFree(ptr);
    }
};
#endif // TEN4_SRC_TEN4_TYPES_H_
