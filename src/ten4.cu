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
///
/// tensorForth kernel - VM dispatcher
/// Note: 1 block per VM, thread 0 active only (wasteful?)
///
__KERN__ void
k_vm_exec(VM *vm) {
    extern __shared__ DU *ss;                   ///< shared mem for ss (much faster)
    
    DU *ss0 = vm->ss.v;                         ///< VM's data stack
    MEMCPY(ss, ss0, sizeof(DU) * T4_SS_SZ);     /// * copy stack into shared memory block
    vm->ss.v = ss;                              /// * redirect data stack to shared memory
    ///
    /// * enter ForthVM outer loop
    /// * Note: single-threaded, dynamic parallelism when needed
    ///
    vm->outer();                                /// * enter VM outer loop
        
    MEMCPY(ss0, ss, sizeof(DU) * T4_SS_SZ);     /// * copy updated stack back to global mem
    vm->ss.v = ss0;                             /// * restore stack ptr
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
         << ", vmss["          << T4_SS_SZ << "*" << VM_COUNT << "]"
         << ", tensor="        << T4_OSTORE_SZ/1024/1024 << "M"
         << endl;
    ///
    /// allocate tensorForth system memory blocks
    ///
    sys = new System(cin, cout, khz, verbose);
}

__HOST__ void
TensorForth::setup() {
    for (int i=0; i < VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        MM_ALLOC(&h->vm, sizeof(VM_TYPE));
//        (h->vm = new VM_TYPE(i, sys))->init();     ///< instantiate VMs
        GPU_ERR(cudaStreamCreate(&h->st));
        GPU_ERR(cudaEventCreate(&h->t0));
        GPU_ERR(cudaEventCreate(&h->t1));
    }
    vm_pool[0].vm->state = HOLD;
}

__HOST__ int
TensorForth::tally() {
    if (sys->trace() <= 1) return 0;
    
    int cnt[4] = { 0, 0, 0, 0};                    /// STOP, HOLD, QUERY, NEST
    for (int i=0; VM_COUNT; i++) {
        cnt[vm_pool[i].vm->state]++;
    }
    cout << "VM.state[STOP,HOLD,QUERY,NEST]=[";
    for (int i = 0; i < 4; i++) cout << " " << cnt[i];
    cout << " ]" << std::endl;
    
#if T4_MMU_DEBUG
    int m0 = (int)sys->mm->here() - 0x80;
    db->mem_dump(cout, m0 < 0 ? 0 : m0, 0x80);
#endif // T4_MMU_DEBUG
        
    return 1;
}

__HOST__ int
TensorForth::run() {
    int n_vm = 1;
    while (n_vm && sys->readline()) {
        n_vm = 0;
        for (int i=0; i<VM_COUNT; i++) {
            VM_Handle *h  = &vm_pool[i];
            VM        *vm = h->vm;
            if (vm->state == STOP) continue;
            n_vm++;
            
            cudaEventRecord(h->t0, h->st);
            k_vm_exec<<<1, 1, T4_SS_SZ, h->st>>>(vm);
            GPU_CHK();
            cudaEventRecord(h->t1, h->st);
            cudaStreamWaitEvent(h->st, h->t1);       // CPU will wait here
            
            float dt;
            cudaEventElapsedTime(&dt, h->t0, h->t1);
            
            switch (vm->state) {
            case HOLD:  VLOG1("%d} VM[%d] HOLD\n", vm->id, vm->id);   break;
#if T4_ENABLE_OBJ                
            case QUERY: if (!vm->compile) db->ss_dump(i, vm->ss.idx); break;
#endif // T4_ENABLE_OBJ                
            }
        }
        sys->flush();             /// * flush output buffer
        tally();                  /// * tally debug info
    }
    return 0;
}

__HOST__ void
TensorForth::teardown(int sig) {
    for (int i=0; i < VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        
        GPU_ERR(cudaEventDestroy(h->t1));
        GPU_ERR(cudaEventDestroy(h->t0));
        GPU_ERR(cudaStreamDestroy(h->st));
    }
}
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


    
