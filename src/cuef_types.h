/*! @file
  @brief
  cueForth macros and internal type definitions

  <pre>
  Copyright (C) 2022- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_CUEF_TYPES_H_
#define CUEF_SRC_CUEF_TYPES_H_

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

// short notation
typedef uint64_t    U64;                    // unsigned integers
typedef uint32_t    U32;
typedef uint16_t    U16;
typedef uint8_t     U8;

typedef int32_t     S32;                    // signed integers
typedef int16_t     S16;
typedef int8_t      S8;

typedef uintptr_t   U32A;                   // pointer address
typedef double      F64;                    // double precision float
typedef float       F32;                    // single precision float

//===============================================================================
// cueForth simple types (non struct)
typedef S32         GI;                     // signed integer
typedef F32         GF;                     // float
typedef U16         GS;                     // symbol index
typedef S32         GP;                     // offset, i.e. object pointer

typedef U16         IU;                     // size of a instruction unit
typedef F32         DU;                     // size of a data unit
#define DU0         0.0f                    /* default data value */

///==============================================================================
///
/// colon word compiler
/// Note:
///   * we separate dict and pmem space to make word uniform in size
///   * if they are combined then can behaves similar to classic Forth
///   * with an addition link field added.
///
enum {
    NOP = 0, DOVAR, DOLIT, DOSTR, DOTSTR, BRAN, ZBRAN, DONEXT, DOES, TOR
} forth_opcode;

// pointer arithmetic, this will not work in multiple segment implementation
#define U8PADD(p, n)    ((U8*)(p) + (n))    // add
#define U8PSUB(p, n)    ((U8*)(p) - (n))    // sub

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
#endif // CUEF_SRC_CUEF_TYPES_H_
