/*! @file
  @brief
  cueForth value definitions non-optimized
*/
#include <iostream>          // cin, cout
#include <signal.h>
using namespace std;

#include "cuef_config.h"
#include "aio.h"             // CUDA async IO
#include "eforth.h"          // eForth core
#include "cueforth.h"        // wrapper

__GPU__	ForthVM *vm_pool[MIN_VM_COUNT];
///
/// instantiate VMs
/// TODO: use shared memory
///
__KERN__ void
cueforth_init(Istream *istr, Ostream *ostr) {
	int i = blockIdx.x;
    if (threadIdx.x!=0) return;

	vm_pool[i] = new ForthVM(istr, ostr); // instantiate VM
	vm_pool[i]->init();                   // initialize dictionary
}
///
/// check VM status
/// TODO: Dynamic Parallel
///
__KERN__ void
cueforth_busy(int *busy) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	*busy = 0;
	for (int i=0; i<MIN_VM_COUNT; i++) {
		if (vm_pool[i]->status == VM_RUN) {
			*busy = 1;
			break;
		}
	}
}
///
///
#include <stdio.h>
__KERN__ void
cueforth_exec() {
	const char *s[] = {"READY", "RUN", "WAITING", "STOPPED"};
    if (threadIdx.x!=0) return;

    ForthVM *vm = vm_pool[blockIdx.x];
    if (vm->status == VM_RUN) vm->outer();
    else 					  printf("VM[%d] %s\n", blockIdx.x, s[vm->status]);
}

CueForth::CueForth(bool trace) {
    aio = new AIO(trace);

    cueforth_init<<<MIN_VM_COUNT, 1>>>(aio->istream(), aio->ostream());
    GPU_CHK();
}
CueForth::~CueForth() {
	delete aio;
	cudaDeviceReset();
}

__HOST__ int
CueForth::is_running() {
	int h_busy;
	int *d_busy = 0;
	cudaMalloc((void**)&d_busy, sizeof(int));
	//LOCK();                 // TODO: lock on vm_pool
	cueforth_busy<<<1, 1>>>(d_busy);
	GPU_SYNC();
	//UNLOCK();               // TODO:

	cudaMemcpy(&h_busy, d_busy, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_busy);

	return h_busy;
}

__HOST__ int
CueForth::run() {
	while (is_running()) {
		if (aio->readline()) {        // feed from host console to managed input buffer
			cueforth_exec<<<1,1>>>(); // TODO: multiple VM destination, shared memory
			GPU_CHK();
			aio->flush();             // flush output buffer
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
void sigsegv_handler(int sig, siginfo_t *si, void *arg) {
	cout << "Exception caught at: " << si->si_addr << endl;
	exit(1);
}

void sigtrap() {
	struct sigaction sa;
	memset(&sa, 0, sizeof(struct sigaction));
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = sigsegv_handler;
	sa.sa_flags     = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, NULL);
}

int main(int argc, char**argv) {
	sigtrap();

    cout << CUEF_VERSION << " init" << endl;
    CueForth *f = new CueForth(CUEF_DEBUG);

    cout << CUEF_VERSION << " start" << endl;
    f->run();

    cout << CUEF_VERSION << " done." << endl;
    f->teardown();
}
