/** -*- c++ -*-
 * @file - ten4.cu
 * @brief - tensorForth value definitions non-optimized
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
#include "ldr/loader.h"      // default dataset loader
#include "nn/netvm.h"        // VM + ForthVM + TensorVM + NetVM
#include "ten4.h"            // wrapper

#define MAJOR_VERSION        "3"
#define MINOR_VERSION        "0"

__GPU__ NetVM *vm_pool[VM_MIN_COUNT]; /// TODO: CC - polymorphic does not work?
///
/// instantiate VMs (threadIdx.x is vm_id)
///
__KERN__ void
k_ten4_init(int khz, Istream *istr, Ostream *ostr, MMU *mmu) {
    auto  g   = cg::this_thread_block();
    int   vid = g.thread_rank();                ///< VM id

    if (vid < VM_MIN_COUNT) {
        NetVM *vm = vm_pool[vid] = new NetVM(khz, istr, ostr, mmu);  /// * instantiate VM
        vm->ss.init(mmu->vmss(vid), T4_SS_SZ);  /// * point data stack to managed memory block
        vm->state = VM_STOP;                    /// * workers wait in queue
        
        if (vid==0) {
            vm->init();                         /// * initialize common dictionary (once only)
            mmu->status();                      /// * report MMU status after init
            vm->state = VM_READY;               /// * VM[0] available for work
        }
    }
    g.sync();
}
///
/// check VM status (using warp-level collectives)
///
__KERN__ void
k_ten4_tally(vm_state *vmst, int *vmst_cnt) {
    const auto g   = cg::this_thread_block();
    const int  vid = g.thread_rank();            ///< VM id
    
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
    CUX(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device));

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
    MM_ALLOC(&vmst, VMST_SZ);                   ///> allocate for state of VMs
    MM_ALLOC(&vmst_cnt, sizeof(int)*4);
    
    Loader::init();
    ///
    /// instantiate virtual machines
    ///
    int t = WARP(VM_MIN_COUNT);                 ///> thread count = 32 modulo
    k_ten4_init<<<1, t>>>(khz, aio->istream(), aio->ostream(), mmu); // create VMs
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
        if (HAS_RUN || aio->readline()) { /// * feed from host console to managed input buffer
            ///
            /// CUDA 11.6, dynamic parallelism does not work with coop-launch
            ///
            k_ten4_exec<<<VM_MIN_COUNT, 1, VMSS_SZ>>>();
            GPU_CHK();
            
            aio->flush();                 /// * flush output buffer
            k_ten4_sweep<<<1, 1>>>(mmu);  /// * release buffered memory
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
    
    const string APP = string(T4_APP_NAME) + " " + MAJOR_VERSION + "." + MINOR_VERSION;
    Options opt;
    opt.parse(argc, argv);
    
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


    
