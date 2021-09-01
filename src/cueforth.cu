/*! @file
  @brief
  cueForth value definitions non-optimized
*/
#include <iostream>
#include "istream.h"
#include "ostream.h"
#include "eforth.h"
#include "cueforth.h"

// forward declaration for implementation
extern "C" __KERN__ void mmu_init(void *ptr, U32 sz);

__GPU__ cuef::ForthVM *vm_pool[MIN_VM_COUNT];
__GPU__ cuef::istream *istr;
__GPU__ cuef::ostream *ostr;

__KERN__ void eforth_init(U8 *ibuf, U8 *obuf) {
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	istr = new cuef::istream((char*)ibuf);
	ostr = new cuef::ostream((char*)obuf);

	for (int i=0; i<MIN_VM_COUNT; i++) {
		vm_pool[i] = new cuef::ForthVM(*istr, *ostr);
		vm_pool[i]->init();                  			 // initialize dictionary
	}
	return;
}

__KERN__ void eforth_exec() {
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	vm_pool[0]->outer();

	return;
}

CueForth::CueForth(istream &in, ostream &out) : cin(in), cout(out) {}
CueForth::~CueForth() {
	if (_obuf) _free(_obuf);
	if (_ibuf) _free(_ibuf);
	if (_heap) _free(_heap);
	cudaDeviceReset();
}

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

	printf("cueForth initializing...");

	_heap = (U8*)_malloc(CUEF_HEAP_SIZE, 1);				// allocate main block (i.e. RAM)
	if (!_heap)  return -10;
	_ibuf = (U8*)_malloc(CUEF_IBUF_SIZE, 1);				// allocate main block (i.e. RAM)
	if (!_ibuf)  return -11;
	_obuf = (U8*)_malloc(CUEF_OBUF_SIZE, 1);				// allocate output buffer
	if (!_obuf)  return -12;

	//mmu_init<<<1,1>>>(mem, CUEF_HEAP_SIZE);				// setup memory management
	eforth_init<<<1,1>>>(_ibuf, _obuf);						// setup basic classes	(TODO: => ROM)
	GPU_SYNC();

    U32 sz0, sz1;
	cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
	cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
	cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

	printf("cueForth initialized, ready to go...");

    return 0;
}

__HOST__ int
CueForth::run()
{
	printf("cuef session starting...");
	// kick up main loop until all VM are done
    string idiom;
	while (cin >> idiom) {
        printf("%s=>", idiom.c_str());
    }

	printf("cuef session completed.");

	return 0;
}

__HOST__ void
CueForth::teardown(int sig) {}

int main(int argc, char**argv) {
	CueForth *f = new CueForth(cin, cout);
	f->setup();
	cout << "cueForth starting..." << endl;
	f->run();
	f->teardown();
    cout << "done!" << endl;
}

