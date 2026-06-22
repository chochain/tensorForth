/**
 * @file
 * @brief tensorForth host macros and internal type definitions
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __TEN4_TYPES_H_
#define __TEN4_TYPES_H_
#pragma  once

#include <cstdio>
#include <cmath>
#include "ten4_config.h"
///
///@name Debug/Tracing options
///@brief INFO/WTRACE/DEBUG/WARN/ERROR
///@{
#define INFO(...)           printf(__VA_ARGS__)
#define ERROR(...)          printf(__VA_ARGS__)
#define NA(msg)             ({ ERROR("method not supported: %s\n", msg); })
///
/// Tracer by verbosity level
///
#if   T4_VERBOSE > 1
#define TRACE(...)          INFO(__VA_ARGS__)
#define DEBUG(...)          INFO(__VA_ARGS__)
#elif T4_VERBOSE > 0
#define TRACE(...)          INFO(__VA_ARGS__)
#define DEBUG(...)
#else  // T4_VERBOSE==0
#define TRACE(...)
#define DEBUG(...)
#endif // T4_VERBOSE
///
/// Memory tracing specific
///
#if MM_DEBUG
#define MM_DB(...)          DEBUG(__VA_ARGS__)
#define NN_DB(...)          TRACE(__VA_ARGS__)
#else  // !MM_DEBUG
#define MM_DB(...)
#define NN_DB(...)
#endif // MM_DEBUG
///@}
namespace t4 {

#define __INLINE__  [[gnu::always_inline]] inline

///@name Alignment macros
///@{
#define ALIGN2(sz)  ((sz) + (sz & 0x1))
#define ALIGN4(sz)  ((sz) + (-(sz) & 0x3))
#define ALIGN8(sz)  ((sz) + (-(sz) & 0x7))
#define ALIGN16(sz) ((sz) + (-(sz) & 0xf))
#define ALIGN(sz)   ALIGN4(sz)
///@}
#define MUTEX_LOCK(p)
#define MUTEX_FREE(p)

#define ASSERT(X) \
    if (!(X)) ERROR("ASSERT: line %d in %s\n", __LINE__, __FILE__);
#define H_ALLOC(p,...)     *((void**)(p)) = std::malloc(__VA_ARGS__)
#define H_FREE(m)          std::free(m)
///
///@name Portable types (Rust alike)
///@{
typedef uint64_t    U64;                    ///< 64-bit unsigned integer
typedef uint32_t    U32;                    ///< 32-bit unsigned integer
typedef uint16_t    U16;                    ///< 16-bit unsigned integer
typedef uint8_t     U8;                     ///< 8-bit  unsigned integer
typedef uintptr_t   UFP;                    ///< function pointer type

typedef int64_t     S64;                    ///< 64-bit signed integer
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
#define DUNIT       0                       /**< data unit 0=F32, 1=F64 */
typedef U32         IU;                     /**< instruction unit       */
typedef F32         DU;                     /**< data unit              */
typedef F64         DU2;                    /**< double preciesion data */
#define DU0         ((DU)0.0f)              /**< default data value 0   */
#define DU1         ((DU)1.0f)              /**< default data value 1   */
#define DU_EPS      ((DU)1.0e-6)            /**< floating point epsilon */
#define ZEQ(d)      (fabsf(d) < DU_EPS)     /**< zero check             */
#define EQ(a,b)     (ZEQ((a) - (b)))        /**< arithmatic equal       */
#define LT(a,b)     (((a) - (b)) < -DU_EPS) /**< arithmatic lesser than */
#define GT(a,b)     (((a) - (b)) > DU_EPS)  /**< arithmatic greater than*/
#define BOOL(d)     (ZEQ(d) ? DU0 : -DU1)   /**< default boolean        */
///
/// data conversion macros
///
/// Note:
///   static_cast<int>(23.5) => 23 (truncate)
///   __float2int_rn(23.5)   => 24 (to round-to-nearest)
///
#define INT(f)      (static_cast<S32>(f))   /**< floor integer -1.99=>-1, -2.01=>-2 */
#define UINT(f)     (static_cast<U32>(f))   /**< unsigned int -1.99=>1, 2.01=>2,    */
#define I2D(i)      (static_cast<DU>(i))    /**< expand int to float                */
#define D2I(f)      (static_cast<S32>(f))   /**< DU to signed integer               */
///@name General Data Types for IO Event
///@{
typedef enum {
    GT_EMPTY = 0,
    GT_INT,
    GT_U32,
    GT_FLOAT,
    GT_STR,
    GT_OBJ,
    GT_FMT,                           ///< output formatting
    GT_OPX,                           ///< complex object ops
    GT_TBX                            ///< tensorboard ops
} GT;
///@}
///@name General Opocode Type for IO Event
///@{
typedef enum {
    OP_FLUSH = 0,                     ///< flush output stream
    OP_DICT,
    OP_WORDS,
    OP_SEE,
    OP_DUMP,
    OP_SS,
    /// tensor, dataset ops
    OP_T2PNG,                         ///< persist tensor to a PNG file
    OP_TSAVE,                         ///< persist tensor (for NumPy, Panda)
    OP_TLOAD,                         ///< load tensor from NumPy dump
    OP_DATA,                          ///< dataset init
    OP_NORM,                          ///< dataset normalization
    OP_FETCH,                         ///< dataset retrieve
    OP_NSAVE,                         ///< network model presistance
    OP_NLOAD,                         ///< network modek restore
} OP;
///@}
///@name TensorBoard Opocode Type
///@{
typedef enum {
    TB_INIT = 0,                      ///< initialize SummaryWriter
    TB_STEP,                          ///< set current step
    TB_SCALAR,                        ///< scalar event
    TB_TEXT,                          ///< text event
    TB_IMAGE,                         ///< image event
    TB_TILE,                          ///< image in tile (10 wide)
    TB_HISTO,                         ///< histogram event
    TB_GRAPH,                         ///< graph event
    TB_EMBED                          ///< embedding projector
} TB_OP;
///@}
///@name IO operators
///@{
typedef enum { CR=0, RDX, DOT, UDOT, EMIT, SPCS } io_op;
///@}
///
/// colon word compiler
/// Note:
///   * we separate dict and pmem space to make word uniform in size
///   * if they are combined then can behaves similar to classic Forth
///   * with an addition link field added.
///
///@name primitive opcode
///{@
typedef enum {
    EXIT=0, NEXT, LOOP, LIT, VAR, STR, DOTQ, BRAN, ZBRAN, FOR,
    DO, KEY, MAX_OP=0xf
} prim_op;
///@}
struct OnHost {
    void *operator new(size_t sz) {
        void *ptr;
        H_ALLOC(&ptr, sz);
        DEBUG("new Host Obj %p size=%ld byes\n", ptr, sz);
        return ptr;
    }
    void operator delete(void *ptr) { H_FREE(ptr); }
};

#ifdef __CUDACC__     // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#include <cuda.h>
#include <cooperative_groups.h>
///
///@name CUDA support macros
///@{
#define __GPU__             __device__
#define __HOST__            __host__
#define __BOTH__            __host__ __device__
#define __KERN__            __global__
#define __INLINE__          __forceinline__
typedef cudaStream_t        STREAM;
typedef cudaEvent_t         EVENT;

//#define MUTEX_LOCK(p)       while (atomicCAS((int *)&p, 0, 1)!=0)
//#define MUTEX_FREE(p)       atomicExch((int *)&p, 0)

#define GPU_SYNC()          cudaDeviceSynchronize()
#define GPU_ERR(c) {             \
    cudaError_t code = (c);      \
    if (code != cudaSuccess) {   \
        ERROR("cudaERROR[%d] %s@%s %d\n", code, cudaGetErrorString(code), __FILE__, __LINE__); \
        cudaDeviceReset();       \
    }}
#define GPU_CHK()          GPU_ERR(cudaDeviceSynchronize())
#define MM_ALLOC(...)      GPU_ERR(cudaMallocManaged(__VA_ARGS__))
#define MM_FREE(m)         GPU_ERR(cudaFree(m))

namespace cg = cooperative_groups;
#define K_RUN(...)         GPU_ERR(cudaLaunchCooperativeKernel(__VA_ARGS__))

#define H2D(dst,src,sz)    GPU_ERR(cudaMemcpy((void*)(dst),(void*)(src),sz,cudaMemcpyHostToDevice))
#define D2H(dst,src,sz)    GPU_ERR(cudaMemcpy((void*)(dst),(void*)(src),sz,cudaMemcpyDeviceToHost))
///@}
#else  // !__CUDACC__
#define __HOST__

#endif // __CUDACC__  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

} // namespace t4

#endif // __TEN4_TYPES_H_
