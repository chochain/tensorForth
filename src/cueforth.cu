/*! @file
  @brief
  cueForth value definitions non-optimized
*/
#include <iostream>          // cin, cout
using namespace std;

#include "cuef_config.h"
#include "aio.h"             // CUDA async IO
#include "eforth.h"          // eForth core
#include "cueforth.h"        // wrapper

// forward declaration for implementation
extern "C" __KERN__ void mmu_init(void *ptr, U32 sz);

__GPU__ __managed__ ForthVM *vm_pool[MIN_VM_COUNT];

__KERN__ void
eforth_init(Istream *istr, Ostream *ostr) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;

    for (int i=0; i<MIN_VM_COUNT; i++) {
        vm_pool[i] = new ForthVM(istr, ostr);     // instantiate new Forth VMs
        vm_pool[i]->init();                       // initialize dictionary
    }
}

__KERN__ void
eforth_exec() {
    if (threadIdx.x!=0) return;

    ForthVM *vm = vm_pool[blockIdx.x];
    while (vm->status == VM_RUN) {
        vm->outer();
    }
}

CueForth::CueForth() {}
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
    case 0:     cudaMalloc(&mem, sz); break;            // allocate device memory
    default:    cudaMallocManaged(&mem, sz);            // managed (i.e. paged) memory
    }
    if (cudaSuccess != cudaGetLastError()) return NULL;

    return mem;
}

__HOST__ void
CueForth::_free(void *mem) {
    cudaFree(mem);
}

__HOST__ int
CueForth::setup(int step, int trace) {
    cudaDeviceReset();

    _heap = (U8*)_malloc(CUEF_HEAP_SIZE, 1);                // allocate main block (i.e. RAM)
    if (!_heap)  return -10;
    _ibuf = (U8*)_malloc(CUEF_IBUF_SIZE, 1);                // allocate main block (i.e. RAM)
    if (!_ibuf)  return -11;
    _obuf = (U8*)_malloc(CUEF_OBUF_SIZE, 1);                // allocate output buffer
    if (!_obuf)  return -12;

    aio = new AIO((char*)_ibuf, (char*)_obuf);

    //mmu_init<<<1,1>>>(mem, CUEF_HEAP_SIZE);               // setup memory management
    eforth_init<<<1,1>>>(aio->istream(), aio->ostream());
    GPU_SYNC();

    U32 sz0, sz1;
    cudaDeviceGetLimit((size_t *)&sz0, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, (size_t)sz0*4);
    cudaDeviceGetLimit((size_t *)&sz1, cudaLimitStackSize);

    return 0;
}

__HOST__ int
CueForth::is_running() {
	int r = 0;
	//LOCK();                 // TODO: lock on vm_pool
	for (int i=0; i<MIN_VM_COUNT; i++) {
		if (vm_pool[i]->status != VM_STOP) r = 1;
	}
	//UNLOCK();               // TODO:
	return r;
}

__HOST__ int
CueForth::run() {
	while (is_running()) {
		cin.getline((char*)_ibuf, CUEF_IBUF_SIZE);
		if (*_ibuf) {
			eforth_exec<<<1,1>>>();
			GPU_SYNC();
			aio->flush();
		}
		yield();
	}
    return 0;
}

__HOST__ void
CueForth::teardown(int sig) {}
///
/// main program
///
int main(int argc, char**argv) {
    CueForth *f = new CueForth();
    cout << CUEF_VERSION << " initializing..." << endl;
    f->setup();

    cout << CUEF_VERSION << " starting..." << endl;
    f->run();

    cout << CUEF_VERSION << " done." << endl;
    f->teardown();
}
