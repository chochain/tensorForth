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

#define MAJOR_VERSION        "1"
#define MINOR_VERSION        "0"

__GPU__    ForthVM *vm_pool[MIN_VM_COUNT];
///
/// instantiate VMs (threadIdx.x is vm_id)
///
__KERN__ void
cueforth_init(Istream *istr, Ostream *ostr, MMU *mmu) {
    int i = threadIdx.x;
    if (i >= MIN_VM_COUNT) return;

    ForthVM *vm = vm_pool[i] = new ForthVM(istr, ostr, mmu);  // instantiate VM
    vm->ss.init(mmu->vss(i), CU4_SS_SZ);   // point data stack to managed memory block

    if (i==0) vm->init();                   // initialize common dictionary (once only)
}
///
/// check VM status (using parallel reduction - overkill?)
///
__KERN__ void
cueforth_busy(int *busy) {
    extern __shared__ bool b[];          // share memory for fast calc

    int i = threadIdx.x;
    b[i] = (i < MIN_VM_COUNT) ? vm_pool[i]->status==VM_RUN : 0;
    __syncthreads();

    for (int n=blockDim.x>>1; n>16; n>>=1) {
        if (i < n) b[i] |= b[i + n];
        __syncthreads();
    }
    if (i < 16) {                        // reduce spinning threads
        b[i] |= b[i + 16];
        b[i] |= b[i + 8];
        b[i] |= b[i + 4];
        b[i] |= b[i + 2];
        b[i] |= b[i + 1];
    }
    if (i==0) *busy = b[0];
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
    DU      *ss  = &shared_ss[b * CU4_SS_SZ];   // adjust stack pointer based on VM id
    DU      *ss0 = vm->ss.v;                    // capture VM data stack
    MEMCPY(ss, ss0, sizeof(DU) * CU4_SS_SZ);    // copy stack into shared memory block
    vm->ss.v = ss;                              // redirect data stack to shared memory

    if (vm->status == VM_RUN) vm->outer();
    else printf("VM[%d] %s\n", blockIdx.x, st[vm->status]);

    __syncthreads();
    MEMCPY(ss0, ss, sizeof(DU) * CU4_SS_SZ);    // copy updated stack to managed memory
    vm->ss.v = ss0;                             // restore stack back to VM
}

CueForth::CueForth(bool trace) {
    mmu = new MMU();
    aio = new AIO(mmu, trace);
    cudaMalloc((void**)&busy, sizeof(int));
    GPU_CHK();

    int t = WARP(MIN_VM_COUNT);                 // thread count = 32 modulo
    cueforth_init<<<1, t>>>(aio->istream(), aio->ostream(), mmu); // init using default stream
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
    int t = WARP(MIN_VM_COUNT);
    cueforth_busy<<<1, t, t * sizeof(bool)>>>(busy);
    GPU_SYNC();
    //UNLOCK();               // TODO:

    cudaMemcpy(&h_busy, busy, sizeof(int), D2H);

    return h_busy;
}

#define VSS_SZ (sizeof(DU)*CU4_SS_SZ*MIN_VM_COUNT)
__HOST__ int
CueForth::run() {
    while (is_running()) {
        if (aio->readline()) {        // feed from host console to managed input buffer
            cueforth_exec<<<1, 1, VSS_SZ>>>();
            GPU_CHK();
            aio->flush();             // flush output buffer
        }
        mmu->mem_dump(cout, 0, 0x40);
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
	string app = string(CU4_APP_NAME) + " " + MAJOR_VERSION + "." + MINOR_VERSION;
    sigtrap();

    cout << app << " init" << endl;
    CueForth *f = new CueForth(CU4_DEBUG);

    cout << app << " start" << endl;
    f->run();

    cout << app << " done." << endl;
    f->teardown();
}
