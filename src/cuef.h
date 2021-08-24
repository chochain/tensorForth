/*! @file
  @brief
  cueForth macros and internal class definitions
*/
#ifndef CUEF_SRC_CUEF_H_
#define CUEF_SRC_CUEF_H_
#include <stdint.h>
#include "cuef_config.h"

#if CUEF_USE_CONSOLE		// use guru local implemented print functions (in puts, sprintf.cu)
#define PRINTF				cuef_printf
#else						// use CUDA printf function
#include <stdio.h>
#define PRINTF				printf
#endif // CUEF_USE_CONSOLE
#define NA(msg)				({ PRINTF("method not supported: %s\n", msg); })

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined(__CUDACC__)

#define __CUEF__ 			__device__
#define __HOST__			__host__
#define __GPU__				__global__
//#define __INLINE__
#define __INLINE__			__forceinline__
#define __UCODE__ 			__CUEF__ void
#define __CFUNC__			__CUEF__ void
#define MUTEX_LOCK(p)  		while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  		atomicExch((int *)&p, 0)

#define ALIGN4(sz)			((sz) + (-(sz) & 0x3))
#define ALIGN8(sz) 			((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)  		((sz) + (-(sz) & 0xf))
#define ASSERT(X) \
	if (!(X)) PRINTF("ASSERT tid %d: line %d in %s\n", threadIdx.x, __LINE__, __FILE__);
#define GPU_SYNC()			{ cudaDeviceSynchronize(); }
#define GPU_CHK()			{ cudaDeviceSynchronize(); ASSERT(cudaGetLastError()==cudaSuccess); }

#else  // defined(__CUDACC__)

#define __CUEF__
#define __INLINE__ 			inline
#define __HOST__
#define __GPU__
#define ALIGN(sz) 			((sz) + (-(sz) & 3))
#define ASSERT(X) 			assert(x)

#endif // defined(__CUDACC__)

#define GT_BOOL(v)		((v) ? GT_TRUE : GT_FALSE)

// short notation
typedef uint64_t    U64;
typedef uint32_t	U32;
typedef uint16_t    U16;
typedef uint8_t		U8;

typedef int32_t     S32;					// signed integer
typedef int16_t		S16;
typedef int8_t		S8;
typedef uintptr_t   U32A;					// pointer address

typedef double		F64;					// double precision float
typedef float       F32;					// single precision float

typedef uint32_t    U32;
typedef uint8_t     U8;

//===============================================================================
// cueForth simple types (non struct)
typedef S32			GI;						// signed integer
typedef F32	 		GF;						// float
typedef U16			GS;						// symbol index
typedef S32			GP;						// offset, i.e. object pointer

#ifdef __cplusplus
}
#endif // __cplusplus

class CuefSession {			    // 16-byte
	istringstream	&stdin;		// input stream
	ostringstream	&stdout;	// output stream
	U16 			id;
	U16 			trace;
	struct RSes 	*next;
};

#endif // CUEF_SRC_CUEF_H_
