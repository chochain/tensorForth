/** -*- c++ -*-
 * @file
 * @brief TensorForth class - tensorForth main driver between CUDA and host
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 * Benchmark: 1K*1K cycles on 3.2GHz AMD, Nvidia GTX1660
 *    + 19.0 msec - REALLY SLOW! Probably due to heavy branch divergence.
 *    + 21.1 msec - without NXT cache in nest() => branch is slow
 *    + 19.1 msec - without push/pop WP         => static ram access is fast
 *    + 20.3 msec - token indirect threading    => not that much worse but portable
 */
#include <iostream>          // cin, cout
#include <signal.h>

using namespace std;
#include "ten4.h"            // wrapper

__GPU__ VM *vm_pool[VM_MIN_COUNT];      ///< polymorphic VM pool
///
/// instantiate VMs (threadIdx.x is vm_id)
///
__KERN__ void
k_ten4_init(Istream *istr, Ostream *ostr, MMU *mmu) {
    int vid = threadIdx.x;
    
    if (vid >= VM_MIN_COUNT) return;    /// * Note: watch for divergence
    
    VM *vm = vm_pool[vid] =             ///< instantiate VMs
        new NetVM(vid, istr, ostr, mmu); 
    
    vm->state = VM_STOP;                /// * workers wait in queue
    if (vid==0) {                       /// * only once 
        vm->init();                     /// * initialize common dictionary
        mmu->status();                  /// * report MMU status after init
        vm->state = VM_READY;           /// * VM[0] available for work
    }
}
///
/// check VM status (using warp-level collectives)
///
__KERN__ void
k_ten4_tally(vm_state *vmst, int *vmst_cnt) {
    const auto g   = cg::this_thread_block();
    const int  vid = g.thread_rank();           ///< VM id
    
    for (int i = 0; i < 4; i++) vmst_cnt[i] = 0;
    g.sync();
    
    if (vid < VM_MIN_COUNT) {
        vm_state *s = &vm_pool[vid]->state;
        if (*s == VM_WAIT) *s = VM_RUN;         ///< assuming WAIT is done
        vmst[vid] = *s;
        atomicAdd(&vmst_cnt[*s], 1);
    }
}
///
/// tensorForth kernel - VM dispatcher
/// Note: 1 block per VM, thread 0 active only (wasteful?)
///
__KERN__ void
k_ten4_exec() {
    extern __shared__ DU shared_ss[];           ///< use shard mem for ss

    const int tx  = threadIdx.x;                ///< thread id (0 only)
    const int vid = blockIdx.x;                 ///< VM id
    
    VM *vm = vm_pool[vid];
    if (tx == 0 && vm->state != VM_STOP) {      /// * one thread per VM
        DU *ss  = &shared_ss[vid * T4_SS_SZ];   ///< each VM uses its own ss
        DU *ss0 = vm->ss.v;                     ///< VM's data stack

        MEMCPY(ss, ss0, sizeof(DU) * T4_SS_SZ); /// * copy stack into shared memory block
        vm->ss.v = ss;                          /// * redirect data stack to shared memory
        ///
        /// * enter ForthVM outer loop
        /// * Note: single-threaded, dynamic parallelism when needed
        ///
        vm->outer();                            /// * enter VM outer loop
        
        MEMCPY(ss0, ss, sizeof(DU) * T4_SS_SZ); /// * copy updated stack back to global mem
        vm->ss.v = ss0;                         /// * restore stack ptr
    }
    /// * grid sync needed but CUDA 11.6 does not support dymanic parallelism
}
///
/// clean up marked free tensors
///
__KERN__ void
k_ten4_sweep(MMU *mmu) {
//    mmu->lock();
    if (blockIdx.x == 0 && threadIdx.x == 0) {  /// * one thread only
        mmu->sweep();
    }
//    mmu->unlock(); !!! DEAD LOCK now
}

__KERN__ void
k_ten4_vm_select(int vid, vm_state st) {
    /// lock
    if (threadIdx.x==0) {
        vm_pool[vid]->state == st;
    }
    /// unlock
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
    GPU_ERR(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device));

    cout << "\\  GPU " << device
         << " initialized at " << khz/1000 << "MHz"
         << ", dict["          << T4_DICT_SZ << "]"
         << ", pmem="          << T4_PMEM_SZ/1024 << "K"
         << ", vmss["          << T4_SS_SZ << "*" << VM_MIN_COUNT << "]"
         << ", tensor="        << T4_OSTORE_SZ/1024/1024 << "M"
         << endl;
    ///
    /// allocate cuda memory blocks
    ///
    mmu = new MMU(khz, verbose);                ///> instantiate memory manager
    aio = new AIO(mmu);                         ///> instantiate async IO manager
    MM_ALLOC(&vmst, VMST_SZ);                   ///> allocate for state of VMs
    MM_ALLOC(&vmst_cnt, sizeof(int)*4);

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
    Loader::init(verbose);
#endif
    ///
    /// instantiate virtual machines
    ///
    int t = WARP(VM_MIN_COUNT);                 ///> thread count = 32 modulo
    k_ten4_init<<<1, t>>>(aio->istream(), aio->ostream(), mmu); // create VMs
    GPU_CHK();
}

TensorForth::~TensorForth() {
    delete aio;
    
    MM_FREE(vmst_cnt);
    MM_FREE(vmst);
    cudaDeviceReset();
}

__HOST__ int
TensorForth::vm_tally() {
    k_ten4_tally<<<1, WARP(VM_MIN_COUNT)>>>(vmst, vmst_cnt);
    GPU_CHK();
    
    if (mmu->trace() > 1) {
        cout << "VM.state[READY,RUN,WAIT,STOP]=[";
        for (int i = 0; i < 4; i++) cout << " " << vmst_cnt[i];
        cout << " ]" << std::endl;
    }
    return 1;
}

#define HAS_BUSY (vmst_cnt[VM_STOP] < VM_MIN_COUNT)
#define HAS_RUN  (vmst_cnt[VM_RUN] > 0)

__HOST__ int
TensorForth::run() {
    while (vm_tally() && HAS_BUSY) {
        if (HAS_RUN || aio->readline(cin)) {  /// * feed from host console to managed buffer
            ///
            /// CUDA 11.6, dynamic parallelism does not work with coop-launch
            ///
            k_ten4_exec<<<VM_MIN_COUNT, 1, VMSS_SZ>>>();
            GPU_CHK();
            
            aio->flush(cout);                 /// * flush output buffer
            k_ten4_sweep<<<1, 1>>>(mmu);      /// * release buffered memory
            GPU_CHK();
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
    
    const string APP = string(T4_APP_NAME) + " " + T4_MAJOR_VER + "." + T4_MINOR_VER;
    Options opt;
    opt.parse(argc, argv);
    
    GPU_ERR(cudaDeviceSetLimit(cudaLimitStackSize, T4_PER_THREAD_STACK));
    // GPU_ERR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16*1024*1024));
    
    if (opt.help) {
        opt.print_usage(std::cout);
        opt.check_devices(std::cout);
        cout << "\nRecommended GPU: " << opt.device_id << std::endl;
        return 0;
    }
    else opt.check_devices(std::cout, false);

    cout << APP << endl;

    TensorForth *f = new TensorForth(opt.device_id, opt.verbose);
    f->run();

    cout << APP << " done." << endl;
    f->teardown();

    return 0;
}


    
