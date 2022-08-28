/** -*- c++ -*-
 * @file - ten4.cu
 * @brief - tensorForth value definitions non-optimized
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 * Benchmark: 1K*1K cycles on 3.2GHz AMD, Nvidia GTX1660
 *    + 19.0 sec - REALLY SLOW! Probably due to heavy branch divergence.
 *    + 21.1 sec - without NXT cache in nest() => branch is slow
 *    + 19.1 sec - without push/pop WP         => static ram access is fast
 *    + 20.3 sec - token indirect threading    => not that much worse but portable
 */
#include <iostream>          // cin, cout
#include <signal.h>
using namespace std;

#include "netvm.h"           // VM + ForthVM + TensorVM + NetVM
#include "ten4.h"            // wrapper

#define MAJOR_VERSION        "3"
#define MINOR_VERSION        "0"

__GPU__ NetVM *vm_pool[VM_MIN_COUNT]; /// TODO: CC - polymorphic does not work?
///
/// instantiate VMs (threadIdx.x is vm_id)
///
__KERN__ void
k_ten4_init(int khz, Istream *istr, Ostream *ostr, MMU *mmu) {
    int k = threadIdx.x;
    NetVM *vm;
    if (k < VM_MIN_COUNT) {
        vm = vm_pool[k] = new NetVM(khz, istr, ostr, mmu);  // instantiate VM
        vm->ss.init(mmu->vmss(k), T4_SS_SZ);  // point data stack to managed memory block
        if (k==0) vm->init();  /// * initialize common dictionary (once only)
    
        vm->status = VM_RUN;
    }
    __syncthreads();
}
///
/// check VM status (using parallel reduction - overkill?)
///
__KERN__ void
k_ten4_busy(int *busy) {
    extern __shared__ bool b[];              // share memory for fast calc

    int k = threadIdx.x;
    b[k] = (k < VM_MIN_COUNT) ? vm_pool[k]->status==VM_RUN : 0;
    __syncthreads();

    for (int n=blockDim.x>>1; n>16; n>>=1) {
        if (k < n) b[k] |= b[k + n];
        __syncthreads();
    }
    if (k < 16) {                        // reduce spinning threads
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) {
            b[k] |= b[k + i];
        }
    }
    if (k==0) *busy = b[0];
}
///
/// tensorForth kernel - VM dispatcher
///
__KERN__ void
k_ten4_exec(int trace) {
    const char *st[] = {"READY", "RUN", "WAITING", "STOPPED"};
    extern __shared__ DU shared_ss[];
    if (threadIdx.x!=0) return;

    int b    = blockIdx.x;
    VM  *vm  = vm_pool[b];
    DU  *ss  = &shared_ss[b * T4_SS_SZ];        // adjust stack pointer based on VM id
    DU  *ss0 = vm->ss.v;                        // capture VM data stack
    MEMCPY(ss, ss0, sizeof(DU) * T4_SS_SZ);     // copy stack into shared memory block
    vm->ss.v = ss;                              // redirect data stack to shared memory

    if (vm->status == VM_RUN) vm->outer();
    else if (trace > 1) INFO("VM[%d] %s\n", blockIdx.x, st[vm->status]);

    __syncthreads();
    MEMCPY(ss0, ss, sizeof(DU) * T4_SS_SZ);     // copy updated stack to managed memory
    vm->ss.v = ss0;                             // restore stack back to VM
}
///
/// clean up marked free tensors
///
__KERN__ void
k_ten4_sweep(MMU *mmu) {
//    mmu->lock();
    if (blockIdx.x ==0 && threadIdx.x == 0) {
        mmu->sweep();
    }
    __syncthreads();
//    mmu->unlock(); !!! DEAD LOCK now
}

TensorForth::TensorForth(int device, int verbose) {
    ///
    /// set active device
    ///
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        cerr << "\nERR: failed to activate GPU " << device << "\n";
        exit(1);
    }
    ///
    /// query GPU shader clock rate
    ///
    int khz = 0;
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device);
    GPU_CHK();

    cout << "\\  GPU " << device
         << " initialized at " << khz/1000 << "MHz"
         << ", dict["          << T4_DICT_SZ << "]"
         << ", vmss["          << T4_SS_SZ << "*" << VM_MIN_COUNT << "]"
         << ", pmem="          << T4_PMEM_SZ/1024 << "K"
         << ", tensor="        << T4_TENSOR_SZ/1024/1024 << "M"
         << endl;
    ///
    /// allocate cuda memory blocks
    ///
    mmu = new MMU(verbose);                     ///> instantiate memory manager
    aio = new AIO(mmu, verbose);                ///> instantiate async IO manager
    cudaMalloc((void**)&busy, sizeof(int));     ///> allocate managed busy flag
    GPU_CHK();
    ///
    /// instantiate virtual machines
    ///
    int t = WARP(VM_MIN_COUNT);                 ///> thread count = 32 modulo
    k_ten4_init<<<1, t>>>(khz, aio->istream(), aio->ostream(), mmu); // create VMs
    GPU_CHK();
}

TensorForth::~TensorForth() {
    delete aio;
    cudaFree(busy);
    cudaDeviceReset();
}

__HOST__ int
TensorForth::is_running() {
    int h_busy;
    //LOCK();                 // TODO: lock on vm_pool
    int t = WARP(VM_MIN_COUNT);
    k_ten4_busy<<<1, t, t * sizeof(bool)>>>(busy);
    GPU_SYNC();
    //UNLOCK();               // TODO:

    cudaMemcpy(&h_busy, busy, sizeof(int), D2H);

    return h_busy;
}

#define VMSS_SZ (sizeof(DU) * T4_SS_SZ * VM_MIN_COUNT)
__HOST__ int
TensorForth::run() {          /// TODO: check ~CUDA/samples/simpleCallback for multi-workload callback
    int trace = mmu->trace();
    while (is_running()) {
        if (aio->readline()) {        // feed from host console to managed input buffer
            k_ten4_exec<<<VM_MIN_COUNT, 1, VMSS_SZ>>>(trace);
            GPU_CHK();                // cudaDeviceSynchronize() and check error
            aio->flush();             // flush output buffer
            k_ten4_sweep<<<1, 1>>>(mmu);
            cudaDeviceSynchronize();
        }
        yield();
        
#if T4_MMU_DEBUG
        int m0 = (int)mmu->here() - 0x80;
        mmu->mem_dump(cout, m0 < 0 ? 0 : m0, 0x80);
#endif // T4_MMU_DEBUG
    }
    return 0;
}

__HOST__ void
TensorForth::teardown(int sig) {}
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

#include "opt.h"
int main(int argc, char**argv) {
    sigtrap();
    
    const string APP = string(T4_APP_NAME) + " " + MAJOR_VERSION + "." + MINOR_VERSION;
    Options opt;
    opt.parse(argc, argv);
    
    if (opt.help) {
        opt.print_usage(std::cout);
        opt.check_devices(std::cout);
        return 0;
    }

    cout << APP << endl;
    
    TensorForth *f = new TensorForth(opt.device_id, opt.verbose);
    f->run();

    cout << APP << " done." << endl;
    f->teardown();

    return 0;
}


    
