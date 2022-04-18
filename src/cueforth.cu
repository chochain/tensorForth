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

__GPU__	__managed__ ForthVM *vm_pool[MIN_VM_COUNT];
///
/// instantiate VMs
/// TODO: use shared memory
///
__KERN__ void
eforth_init(Istream *istr, Ostream *ostr) {
	int i = blockIdx.x;
    if (threadIdx.x!=0) return;

    vm_pool[i] = new ForthVM(istr, ostr); // instantiate VM
    vm_pool[i]->init();                   // initialize dictionary
}
///
///
#include <stdio.h>
__KERN__ void
eforth_exec() {
	const char *s[] = {"READY", "RUN", "WAITING", "STOPPED"};
	int i = blockIdx.x;
    if (threadIdx.x!=0) return;

    ForthVM *vm = vm_pool[i];
    if (vm->status == VM_RUN) vm->outer();
    else printf("VM[%d] %s\n", i, s[vm->status]);
}

CueForth::CueForth(bool trace) {
    cudaDeviceReset();

    aio = new AIO(trace);

    //mmu_init<<<1,1>>>(mem, CUEF_HEAP_SIZE);               // setup memory management
    eforth_init<<<MIN_VM_COUNT, 1>>>(aio->istream(), aio->ostream());
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
		int st;
		cudaMemcpy(&st, &vm_pool[i]->status, sizeof(int), cudaMemcpyDeviceToHost);
		if (st != VM_STOP) r = 1;
	}
	//UNLOCK();               // TODO:
	GPU_SYNC();
	return r;
}

__HOST__ int
CueForth::run() {
	int i = 4;
	while (i--) {
		//printf("run=%d\n", is_running());
		if (aio->readline()) {      // feed from host console to managed input buffer
			eforth_exec<<<1,1>>>(); // TODO: multiple VM destination, shared memory
			GPU_CHK();
			aio->flush();           // flush output buffer
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
    CueForth *f = new CueForth(CUEF_DEBUG);

    cout << CUEF_VERSION << " start" << endl;
    f->run();

    cout << CUEF_VERSION << " done." << endl;
    f->teardown();
}
