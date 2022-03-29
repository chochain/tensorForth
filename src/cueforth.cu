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
        //vm_pool[i]->init();                       // initialize dictionary
    }
}

#include <stdio.h>
__KERN__ void
eforth_exec() {
    if (threadIdx.x!=0) return;

    vm_pool[0]->outer();
    return;
/*
    ForthVM *vm = vm_pool[blockIdx.x];
    while (vm->status == VM_RUN) {
        vm->outer();
    }
*/
}

CueForth::CueForth(bool trace) {
    cudaDeviceReset();

    aio = new AIO(trace);

    //mmu_init<<<1,1>>>(mem, CUEF_HEAP_SIZE);               // setup memory management
    eforth_init<<<1,1>>>(aio->istream(), aio->ostream());
    GPU_CHK();
}
CueForth::~CueForth() {
	delete aio;
	cudaDeviceReset();
}

__HOST__ int
CueForth::is_running() {
	int r = 0;
	GPU_SYNC();
	//LOCK();                 // TODO: lock on vm_pool
	for (int i=0; i<MIN_VM_COUNT; i++) {
		if (vm_pool[i]->status != VM_STOP) r = 1;
	}
	//UNLOCK();               // TODO:
	GPU_SYNC();
	return r;
}

__HOST__ int
CueForth::run() {
	//int rr = is_running();
	int i = 4;
	while (--i) {
		if (aio->readline()) {
			eforth_exec<<<1,1>>>();
			GPU_CHK();
			//aio->flush();
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
    cout << CUEF_VERSION << " init" << endl;
    CueForth *f = new CueForth(true);

    cout << CUEF_VERSION << " start" << endl;
    f->run();

    cout << CUEF_VERSION << " done." << endl;
    f->teardown();
}
