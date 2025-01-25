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
    extern __shared__ DU *ss[];                 ///< shared mem for ss (much faster)
    
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
         << ", vmss["          << T4_SS_SZ << "*" << VM_MIN_COUNT << "]"
         << ", tensor="        << T4_OSTORE_SZ/1024/1024 << "M"
         << endl;
    ///
    /// allocate tensorForth system memory blocks
    ///
    sys = new System(khz, verbose);
}

__HOST__ void
TensorForth::setup() {
    for (int i=0; i < VM_MIN_COUNT; i++) {
        T4Entry *e = &vm_pool[i];
        (e->vm = new VM_TYPE(vid, sys))->init();     ///< instantiate VMs
        GPU_ERR(cudaCreateStream(&e->st));
        GPU_ERR(cudaEventCreate(&e->t0));
        GPU_ERR(cudaEventCreate(&e->t1));
    }
    
    vm_pool[0].vm->state = HOLD;
}

__HOST__ int
TensorForth::vm_tally() {
    if (sys->trace() <= 1) return 0;
    
    int cnt[VM_MIN_COUNT] = { 0, 0, 0, 0};
    for (int i=0; VM_MIN_COUNT; i++) {
        cnt[vm_pool[i].vm->state]++;
    }
    cout << "VM.state[STOP,HOLD,QUERY,NEST]=[";
    for (int i = 0; i < 4; i++) cout << " " << cnt[i];
    cout << " ]" << std::endl;
    
    return 1;
}

__HOST__ int
TensorForth::run() {
    int n_vm = 1;
    while (n_vm && sys->readline()) {
        n_vm = 0;
        for (int i=0; i<VM_MIN_COUNT; i++) {
            T4Entry *e  = &vm_pool[i];
            VM      *vm = e->vm;
            if (vm->state == STOP) continue;
            n_vm++;
            cudaEventRecord(e->t0, e->st);
            k_vm_exec<<<1, 1, T4_SS_SZ, e->st>>>(vm);
            GPU_CHK();
            cudaEventRecord(e->t1, e->st);
            cudaStreamWaitEvent(e->t1);       // CPU will wait here
            
            float dt;
            cudaEventElapsedTime(&dt, e->t0, e->t1);
            
            switch (vm->state) {
            case VM_WAIT:  VLOG1("%d} VM[%d] wait\n", vm->vid, vm->vid); break;
            case VM_QUERY: if (!vm->compile) sys.ss_dump(*vm);           break;
            }
        }
        aio->flush(cout);     /// * flush output buffer
///
/// clean up marked free tensors
///
        sys->mmu_sweep();     /// * release buffered memory
        
#if T4_MMU_DEBUG
        int m0 = (int)sys->mmu.here() - 0x80;
        sys->mem_dump(cout, m0 < 0 ? 0 : m0, 0x80);
#endif // T4_MMU_DEBUG
        
        tally();
    }
    return 0;
}

__HOST__ void
TensorForth::teardown(int sig) {
    for (int i=0; i < VM_MIN_COUNT; i++) {
        T4Entry *e = &pool[i];
        
        GPU_ERR(cudaDestroyStream(e->st));
        GPU_ERR(cudaDestroyEvent(p->t0));
        GPU_ERR(cudaDestroyStream(p->t1));
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


    
