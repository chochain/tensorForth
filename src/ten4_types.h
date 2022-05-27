/*! @file
  @brief
  tensorForth macros and internal type definitions

  <pre>
  Copyright (C) 2022- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef TEN4_SRC_TEN4_TYPES_H_
#define TEN4_SRC_TEN4_TYPES_H_

#define ALIGN4(sz)          ((sz) + (-(sz) & 0x3))
#define ALIGN8(sz)          ((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)         ((sz) + (-(sz) & 0xf))

#if defined(__CUDACC__)
#include <cuda.h>

#define __GPU__             __device__
#define __HOST__            __host__
#define __KERN__            __global__
#define __INLINE__          __forceinline__
    
#define MUTEX_LOCK(p)       while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)       atomicExch((int *)&p, 0)

#define ALIGN(sz)           ALIGN4(sz)
#define PRINTF              printf
#define NA(msg)             ({ PRINTF("method not supported: %s\n", msg); })
#define ASSERT(X) \
    if (!(X)) PRINTF("ASSERT tid %d: line %d in %s\n", threadIdx.x, __LINE__, __FILE__);
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
#define PRINTF              printf
#define NA(msg)             ({ PRINTF("method not supported: %s\n", msg); })
#define ASSERT(X)           assert(x)

#endif // defined(__CUDACC__)
///
/// short-hand for common data types
///
typedef uint64_t    U64;                    /// 64-bit unsigned integer
typedef uint32_t    U32;                    /// 32-bit unsigned integer
typedef uint16_t    U16;                    /// 16-bit unsigned integer
typedef uint8_t     U8;                     /// 8-bit  unsigned integer
typedef uintptr_t   UFP;                    /// function pointer type

typedef int32_t     S32;                    // 32-bit signed integer
typedef int16_t     S16;                    // 16-bit signed integer

typedef double      F64;                    // double precision float
typedef float       F32;                    // single precision float
//===============================================================================
// tensorForth common data types
//
typedef U16         IU;                     /// instruction unit
typedef F32         DU;                     /// data unit
typedef F64         DU2;                    /// double preciesion data unit
#define DU0         0                       /* default data value     */
#define DU_EPS      1.0e-6                  /* floating point epsilon */
//
// tensorForth generalized data object types
//
typedef S32         GI;                     /// signed integer
typedef F32         GF;                     /// float
typedef U16         GS;                     /// symbol index
typedef S32         GP;                     /// offset, i.e. object pointer
//
// tensorForth complex data object
//
typedef struct {
	union {
		F32 f = 0.0f;
		struct {
			U32 r: 1;       // tensor rank >= 1
            U32 p: 31;      // 2^31 slots
		};
	};
} XU;
///==============================================================================
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