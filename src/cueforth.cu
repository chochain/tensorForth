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

__GPU__    ForthVM *vm_pool[MIN_VM_COUNT];
///
/// instantiate VMs
/// TODO: use shared memory
///
__KERN__ void
cueforth_init(Istream *istr, Ostream *ostr, MMU *mmu) {
    if (threadIdx.x!=0) return;

    int b = blockIdx.x;
    ForthVM *vm = vm_pool[b] = new ForthVM(istr, ostr, mmu);  // instantiate VM
    vm->ss.v = mmu->vss(b);              // point data stack to managed memory block

    if (b==0) vm->init();                // initialize common dictionary (once only)
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
    const char *st[] = {"READY", "RUN", "WAITING", "STOPPED"};
    extern __shared__ DU shared_ss[];
    if (threadIdx.x!=0) return;

    int      b   = blockIdx.x;
    ForthVM *vm  = vm_pool[b];
    DU      *ss  = &shared_ss[b * CUEF_SS_SZ];  // adjust stack pointer based on VM id
    DU      *ss0 = vm->ss.v;                    // VM data stack in managed memory
    MEMCPY(ss, ss0, sizeof(DU) * CUEF_SS_SZ);   // copy stack into shared memory block
    vm->ss.v = ss;                              // redirect data stack to shared memory

    if (vm->status == VM_RUN) vm->outer();
    else printf("VM[%d] %s\n", blockIdx.x, st[vm->status]);

    __syncthreads();
    MEMCPY(ss0, ss, sizeof(DU) * CUEF_SS_SZ);    // copy updated stack to managed memory
    vm->ss.v = ss0;                              // reset stack back to VM
}

CueForth::CueForth(bool trace) {
    mmu = new MMU();
    aio = new AIO(mmu, trace);
    cudaMalloc((void**)&busy, sizeof(int));
    GPU_CHK();

    cueforth_init<<<MIN_VM_COUNT, 1>>>(aio->istream(), aio->ostream(), mmu);
    GPU_CHK();

    //dict->dump(cout, 0, 120*0x10);            // dump memory from host
    //dict->words(cout);                        // dump dictionary from host
}
CueForth::~CueForth() {
    delete aio;
    cudaFree(busy);
    cudaDeviceReset();
}

__HOST__ int
CueForth::is_running() {
    int h_busy;
    //LOCK();                 // TODO: lock on vm_pool
    cueforth_busy<<<1, 1>>>(busy);
    GPU_SYNC();
    //UNLOCK();               // TODO:

    cudaMemcpy(&h_busy, busy, sizeof(int), D2H);

    return h_busy;
}

#define VSS_SZ (sizeof(DU)*CUEF_SS_SZ*MIN_VM_COUNT)
__HOST__ int
CueForth::run() {
    while (is_running()) {
        if (aio->readline()) {        // feed from host console to managed input buffer
            cueforth_exec<<<1, 1, VSS_SZ>>>();
            GPU_CHK();
            aio->flush();             // flush output buffer
        }
        mmu->dump(cout, 0, 0x40);
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
