/**
 * @file
 * @brief tensorForth macros and internal type definitions
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __TEN4_TYPES_H_
#define __TEN4_TYPES_H_
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
///@name CUDA support macros
///@{
#if defined(__CUDACC__)     // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

#include <cuda.h>
#include <cooperative_groups.h>
#define __GPU__             __device__
#define __HOST__            __host__
#define __BOTH__            __host__ __device__
#define __KERN__            __global__
#define __INLINE__          __forceinline__
typedef cudaStream_t        STREAM;
typedef cudaEvent_t         EVENT;

#define MUTEX_LOCK(p)       while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)       atomicExch((int *)&p, 0)

#define ASSERT(X) \
    if (!(X)) ERROR("ASSERT tid %d: line %d in %s\n", threadIdx.x, __LINE__, __FILE__);
#define GPU_SYNC() { cudaDeviceSynchronize(); }
#define CDP_SYNC() { cudaDeviceSynchronize(); }      /** for CDP1 retention */
#define GPU_ERR(c) {             \
    cudaError_t code = (c);      \
    if (code != cudaSuccess) {   \
        ERROR("cudaERROR[%d] %s@%s %d\n", code, cudaGetErrorString(code), __FILE__, __LINE__); \
        cudaDeviceReset();       \
    }}
#define GPU_CHK() {              \
    GPU_SYNC();                  \
    GPU_ERR(cudaGetLastError()); \
    }
#define MM_ALLOC(...)      GPU_ERR(cudaMallocManaged(__VA_ARGS__))
#define MM_FREE(m)         GPU_ERR(cudaFree(m))

namespace cg = cooperative_groups;
#define K_RUN(...)         GPU_ERR(cudaLaunchCooperativeKernel(__VA_ARGS__))

#else  // defined(__CUDACC__)  ===============================================

#define __GPU__
#define __HOST__
#define __KERN__
#define __INLINE__          inline
typedef int                 STREAM;
typedef int                 EVENT;

#define ASSERT(X)           assert(x)

#endif // defined(__CUDACC__)  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

typedef int64_t     S64;                    ///< 64-bit signed integer
typedef int32_t     S32;                    ///< 32-bit signed integer
typedef int16_t     S16;                    ///< 16-bit signed integer

typedef double      F64;                    ///< double precision float
typedef float       F32;                    ///< single precision float
///@}
///@name CUDA specific macros
///@note: consider use of fn<<<_g,_b,0,cudaStreamTailLaunch>>>(...)
///@{
#define ALIGN(sz) ALIGN4(sz)
#define FORK(fn,n,...) {                                    \
    const dim3 _b(T4_DIM_SQ, 1, 1);                         \
    const dim3 _g(((n) + _b.x - 1) / _b.x, 1, 1);           \
    fn<<<_g,_b>>>(__VA_ARGS__,n);                           \
}
#define FORK1(fn,n,...) {                                   \
    fn<<<1,T4_DIM_SQ>>>(__VA_ARGS__,n);                     \
}
#define FORK3(fn,h,w,c,...) {                               \
    const dim3 _b(T4_DIM_SZ, T4_DIM_SZ, 1);                 \
    const dim3 _g(((w) + _b.x - 1) / _b.x,                  \
                  ((h) + _b.y - 1) / _b.y, c);              \
    fn<<<_g,_b>>>(__VA_ARGS__,h,w);                         \
}
#define FORK4(fn,...) { /** N,H,W,C (default params) */     \
    const dim3 _b(T4_DIM_SQ, 1, 1);                         \
    const dim3 _g(((U64)(W)*(H) + _b.x - 1) / _b.x, C, N);  \
    fn<<<_g,_b>>>(__VA_ARGS__);                             \
}
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
#define DU0         ((DU)0.0)               /**< default data value 0   */
#define DU1         ((DU)1.0)               /**< default data value 1   */
#define DU_EPS      ((DU)1.0e-6)            /**< floating point epsilon */
#define ZEQ(d)      (ABS(d) < DU_EPS)       /**< zero check             */
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
#define INT(f)      (static_cast<S32>(f))            /**< floor integer -1.99=>-1, -2.01=>-2 */
#define UINT(f)     (static_cast<U32>(f))            /**< unsigned int -1.99=>1, 2.01=>2,    */
#define I2D(i)      (static_cast<DU>(i))             /**< expand int to float                */
#define D2I(f)      (__float2int_rn(f))              /**< nearest-even int 1.99=>2, 2.01=>2  */
///
/// object classification macros
///
constexpr U32 T4_TYPE_MSK = 0x00000003;              /**< obj view flag  */
constexpr U32 T4_TT_OBJ   = 0x00000001;              /**< data unit flag */
constexpr U32 T4_TT_VIEW  = 0x00000003;              /**< view of object */
constexpr U32 EXT_FLAG    = 0x80000000;              /**< extention flag */
#define DU2X(v)     (*(U32*)&(v))                    /**< to U32 ptr     */
#define SCALAR(v)   ((DU2X(v) &= ~T4_TT_OBJ), (v))   /**< set DU flag    */

#if T4_DO_OBJ
#define IS_OBJ(v)   ((DU2X(v) & T4_TT_OBJ)!=0)             /**< if is an obj   */
#define IS_VIEW(v)  ((DU2X(v) & T4_TYPE_MSK)==T4_TT_VIEW)
#define AS_VIEW(v)  ((DU2X(v) |= T4_TT_VIEW), (v))
#else  // !T4_DO_OBJ
#define IS_OBJ(v)   (0)
#define IS_VIEW(v)  (0)
#define AS_VIEW(v)  (0)
#endif // T4_DO_OBJ
///@}
///>name General Data Types for IO Event
///@{
typedef enum {
    GT_EMPTY = 0,
    GT_INT,
    GT_U32,
    GT_FLOAT,
    GT_STR,
    GT_OBJ,
    GT_FMT,
    GT_OPX
} GT;
///@}
///>name General Opocode Type for IO Event
///@{
typedef enum {
    OP_DICT = 0,
    OP_WORDS,
    OP_SEE,
    OP_DUMP,
    OP_SS,
    OP_TSAVE,
    OP_DATA,
    OP_FETCH,
    OP_NSAVE,
    OP_NLOAD
} OP;
///@}
///>name File Access Mode for IO Event
///@{
typedef std::istream h_istr;          ///< host input stream
typedef std::ostream h_ostr;          ///< host output ostream

typedef enum {
    FAM_WO  = 0,
    FAM_RW  = 1,
    FAM_RAW = 2
} FAM;
///@}
///>name IO Event
typedef struct {
    U32 gt : 4;     // 16 io event types
    U32 sz : 28;    // max 256M payload
    U8  data[];     // data array
} io_event;

#define EVENT_HDR  sizeof(U32)
///@}
///>name Random Number Generator
///@{
typedef enum {
    UNIFORM = 0,
    NORMAL
} rand_opt;
///@}
///>name IO operators
///@{
typedef enum { RDX=0, CR, DOT, UDOT, EMIT, SPCS } io_op;
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
struct Managed {
    void *operator new(size_t sz) {
        void *ptr;
        MM_ALLOC(&ptr, sz);
        DEBUG("new Managed Obj %p size=%ld byes\n", ptr, sz);
        return ptr;
    }
    void operator delete(void *ptr) { MM_FREE(ptr); }
};
#endif // __TEN4_TYPES_H_
