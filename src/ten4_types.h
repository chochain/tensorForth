/**
 * @file
 * @brief tensorForth macros and internal type definitions
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TEN4_TYPES_H_
#define TEN4_SRC_TEN4_TYPES_H_
///
///@name CUDA support macros
///@{
#if defined(__CUDACC__)
#include <cuda.h>
#define __GPU__             __device__
#define __HOST__            __host__
#define __BOTH__            __host__ __device__
#define __KERN__            __global__
#define __INLINE__          __forceinline__

#define MUTEX_LOCK(p)       while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)       atomicExch((int *)&p, 0)

#define NA(msg)             ({ INFO("method not supported: %s\n", msg); })
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
#else  // defined(__CUDACC__)
#define __GPU__
#define __HOST__
#define __KERN__
#define __INLINE__          inline
#define ALIGN(sz)           ALIGN4(sz)
#define PRINTF(fmt,...)     printf(fmt,__VA_ARGS__)
#define NA(msg)             ({ PRINTF("method not supported: %s\n", msg); })
#define ASSERT(X)           assert(x)
#endif // defined(__CUDACC__)
///@}
///@name Debug tracing options
///@{
#if T4_VERBOSE
#define INFO(...)           printf(__VA_ARGS__)
#else  // T4_VERBOSE
#define INFO(...)
#endif // T4_VERBOSE
#if MMU_DEBUG
#define DEBUG(...)          printf(__VA_ARGS__)
#else  // MMU_DEBUG
#define DEBUG(...)
#endif // MMU_DEBUG
#define ERROR(...)          printf(__VA_ARGS__)
///@}
///@name Portable types
///@{
typedef uint64_t    U64;                    ///< 64-bit unsigned integer
typedef uint32_t    U32;                    ///< 32-bit unsigned integer
typedef uint16_t    U16;                    ///< 16-bit unsigned integer
typedef uint8_t     U8;                     ///< 8-bit  unsigned integer
typedef uintptr_t   UFP;                    ///< function pointer type

typedef int32_t     S32;                    ///< 32-bit signed integer
typedef int16_t     S16;                    ///< 16-bit signed integer

typedef double      F64;                    ///< double precision float
typedef float       F32;                    ///< single precision float
///@}
//===============================================================================
/// tensorForth common data types
///
///@name Forth instruction and data types
///@{
typedef U16         IU;                     ///< instruction unit
typedef F32         DU;                     ///< data unit
typedef F64         DU2;                    ///< double preciesion data unit
#define DU0         0                       /**< default data value     */
#define DU_EPS      1.0e-6                  /**< floating point epsilon */
///
/// macros for Tensor definitions
///
#define T4_TENSOR    1                          /**< tensor attibute flag    */
#define IS_TENSOR(d) ((*(U32*)&d) & T4_TENSOR)  /**< check if DU is a tensor */
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

class Managed {
public:
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
