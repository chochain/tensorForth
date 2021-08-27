/*! @file
  @brief
  cueForth value definitions non-optimized
*/
#include <iostream>
#include "cuef.h"

// forward declaration for implementation
extern "C" __KERN__ void mmu_init(void *ptr, U32 sz);
extern "C" __KERN__ void eforth_init(U8 *cin, U8 *cout);

__KERN__ void eforth_init(U8 *cin, U8 *cout) {
	//if (threadId.x!=0 || blockId.x!=0) return;
    //string cmd;
	return;
}

CueForth::CueForth(istream &in, ostream &out) : cin(in), cout(out) {}

__HOST__ void*
CueForth::_malloc(int sz, int type)
{
	void *mem;

	// TODO: to add texture memory
	switch (type) {
	case 0: 	cudaMalloc(&mem, sz); break;			// allocate device memory
	default: 	cudaMallocManaged(&mem, sz);			// managed (i.e. paged) memory
	}
    if (cudaSuccess != cudaGetLastError()) return NULL;

    return mem;
}

__HOST__ void
CueForth::_free(void *mem) {
	cudaFree(mem);
}

__HOST__ int
CueForth::setup(int step, int trace)
{
	cudaDeviceReset();

	PRINTF("cueForth initializing...");

	heap = (U8*)_malloc(CUEF_HEAP_SIZE, 1);					// allocate main block (i.e. RAM)
	if (!heap)  return -10;
	ibuf = (U8*)_malloc(CUEF_IBUF_SIZE, 1);					// allocate main block (i.e. RAM)
	if (!ibuf)  return -11;
	obuf = (U8*)_malloc(CUEF_OBUF_SIZE, 1);					// allocate output buffer
	if (!obuf)  return -12;

	//mmu_init<<<1,1>>>(mem, CUEF_HEAP_SIZE);				// setup memory management
	eforth_init<<<1,1>>>(ibuf, obuf);						// setup basic classes	(TODO: => ROM)
	GPU_SYNC();

    U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	PRINTF("cueForth initialized, ready to go...");

    return 0;
}

__HOST__ int
CueForth::run()
{
	PRINTF("cuef session starting...");
	// kick up main loop until all VM are done
    string idiom;
	while (cin >> idiom) {
        printf("%s=>", idiom.c_str());
    }

	PRINTF("cuef session completed.");

	return 0;
}

__HOST__ void
CueForth::teardown(int sig)
{
	if (obuf) _free(obuf);
	if (ibuf) _free(ibuf);
	if (heap) _free(heap);
	cudaDeviceReset();
}

int main(int argc, char**argv) {
	CueForth *f = new CueForth(cin, cout);
	f->setup();
	cout << "cueForth starting..." << ENDL;
	f->run();
	f->teardown();
    cout << "done!" << ENDL;
}

