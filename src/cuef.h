/*! @file
  @brief
  cueForth macros and internal class definitions
*/
#ifndef CUEF_SRC_CUEF_H_
#define CUEF_SRC_CUEF_H_
//#include <cstdio>
//#include <cstdint>
#include <cuda.h>
#include <sstream>
#include "cuef_config.h"

using namespace std;

#define PRINTF				printf
#define NA(msg)				({ PRINTF("method not supported: %s\n", msg); })

#if defined(__CUDACC__)

#define __GPU__ 			__device__
#define __HOST__			__host__
#define __KERN__			__global__
#define __INLINE__			__forceinline__
    
#define MUTEX_LOCK(p)  		while (atomicCAS((int *)&p, 0, 1)!=0)
#define MUTEX_FREE(p)  		atomicExch((int *)&p, 0)

#define ASSERT(X) \
	if (!(X)) PRINTF("ASSERT tid %d: line %d in %s\n", threadIdx.x, __LINE__, __FILE__);
#define GPU_SYNC()			{ cudaDeviceSynchronize(); }
#define GPU_CHK()			{ cudaDeviceSynchronize(); ASSERT(cudaGetLastError()==cudaSuccess); }

#else  // defined(__CUDACC__)

#define __GPU__
#define __KERN__
#define __HOST__
#define __INLINE__ 			inline
#define ALIGN(sz) 			((sz) + (-(sz) & 3))
#define ASSERT(X) 			assert(x)

#endif // defined(__CUDACC__)

// short notation
typedef uint64_t    U64;                    // unsigned integers
typedef uint32_t	U32;
typedef uint16_t    U16;
typedef uint8_t		U8;

typedef int32_t     S32;					// signed integers
typedef int16_t		S16;
typedef int8_t		S8;

typedef uintptr_t   U32A;					// pointer address
typedef double		F64;					// double precision float
typedef float       F32;					// single precision float

//===============================================================================
// cueForth simple types (non struct)
typedef S32			GI;						// signed integer
typedef F32	 		GF;						// float
typedef U16			GS;						// symbol index
typedef S32			GP;						// offset, i.e. object pointer

// pointer arithmetic, this will not work in multiple segment implementation
#define U8PADD(p, n)	((U8*)(p) + (n))	// add
#define U8PSUB(p, n)	((U8*)(p) - (n))	// sub

class CueForth {
	istream &cin;
	ostream &cout;

	U8 *heap;
	U8 *ibuf;
	U8 *obuf;

    __HOST__ void* _malloc(int sz, int type);
    __HOST__ void  _free(void *mem);

public:
    CueForth(istream &in, ostream &out);
    __HOST__ int   setup(int step=0, int trace=0);
    __HOST__ int   run();
    __HOST__ void  teardown(int sig=0);
};
#endif // CUEF_SRC_CUEF_H_
