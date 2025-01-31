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
#include "ten4.h"            // wrapper

using namespace std;
///
/// tensorForth kernel - VM dispatcher
/// Note: 1 block per VM, thread 0 active only (wasteful?)
///
__GPU__ VM *d_vm_pool[VM_COUNT];

__KERN__ void
k_vm_init(System *sys, VM_Handle *pool) {
    int id = threadIdx.x;
    if (id >= VM_COUNT) return;
    
    VM *vm = pool[id].vm = new VM_TYPE(id, sys);
    vm->init();
    vm->state = id==0 ? HOLD : STOP;

    if (id==0) sys->mu->status();
}

__KERN__ void
k_vm_done(VM_Handle *pool) {
    int id = threadIdx.x;
    if (id >= VM_COUNT) return;
    
    delete pool[id].vm;
}

__KERN__ void
k_vm_exec(VM *vm) {
    __shared__ DU ss[T4_SS_SZ];      ///< shared mem for ss, rs (much faster)
    __shared__ DU rs[T4_RS_SZ];      ///< CC: make sure SZ < 32
    
    bool i   = threadIdx.x;
    bool t0  = i == 0 && blockIdx.x == 0;
    DU   *s0 = vm->ss.v;
    DU   *r0 = vm->rs.v;

    ///> copy stacks from global to shared mem
    for (int n = 0; n < T4_SS_SZ; n += WARP_SZ)
        if (i < vm->ss.idx) ss[n + i] = s0[n + i];
    for (int n = 0; n < T4_RS_SZ; n += WARP_SZ)
        if (i < vm->rs.idx) rs[n + i] = r0[n + i];
    __syncthreads();
    
    if (t0) {
        vm->ss.v = ss;
        vm->rs.v = rs;
        ///
        /// * enter ForthVM outer loop
        /// * Note: single-threaded, dynamic parallelism when needed
        ///
//    vm->outer();                               /// * enter VM outer loop
        /*
        DU ss0 = vm->ss[0];
        for (int i=0; i<10; i++) {
            vm->ss[0] = (DU)i;
        }
        vm->ss[0] = ss0;
        */
    }
    __syncthreads();
    
    ///> copy updated stacks back to global mem
    for (int n = 0; n < T4_SS_SZ; n += WARP_SZ)
        if (i < vm->ss.idx) s0[n + i] = ss[n + i];
    for (int n = 0; n < T4_SS_SZ; n += WARP_SZ)
        if (i < vm->rs.idx) r0[n + i] = rs[n + i];

    if (t0) {
        vm->ss.v = s0;                           /// * restore stack pointers
        vm->rs.v = r0;
    }
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

    cout << "\\ GPU "  << device
         << " at "     << khz/1000 << "MHz"
         << ", dict["  << T4_DICT_SZ << "]"
         << ", pmem="  << T4_PMEM_SZ/1024 << "K"
         << ", ostor=" << T4_OSTORE_SZ/1024/1024 << "M"
         << ", vmss["  << T4_SS_SZ << "*" << VM_COUNT << "]"
         << ", vmrs["  << T4_RS_SZ << "*" << VM_COUNT << "]"
         << endl;
    ///
    /// allocate tensorForth system memory blocks
    ///
    sys = new System(cin, cout, khz, T4_VERBOSE);
    ///
    /// allocate VM handle pool
    ///
    MM_ALLOC(&vm_pool, sizeof(VM_Handle) * VM_COUNT);
}

__HOST__ void
TensorForth::setup() {
    for (int i=0; i < VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        GPU_ERR(cudaStreamCreate(&h->st));          /// * allocate stream
        GPU_ERR(cudaEventCreate(&h->t0));           /// * allocate timers
        GPU_ERR(cudaEventCreate(&h->t1));
    }
    k_vm_init<<<1, WARP(VM_COUNT)>>>(sys, vm_pool); /// * initialize all VMs
    GPU_CHK();
}

__HOST__ int
TensorForth::tally() {
    if (sys->trace() <= 1) return 0;
    
    int cnt[4] = { 0, 0, 0, 0};                    /// STOP, HOLD, QUERY, NEST
    for (int i=0; i < VM_COUNT; i++) {
        cnt[vm_pool[i].vm->state]++;
    }
    cout << "VM.state[STOP,HOLD,QUERY,NEST]=[";
    for (int i = 0; i < 4; i++) cout << " " << cnt[i];
    cout << " ]" << std::endl;
    
#if T4_VERBOSE > 1
    int m0 = (int)sys->mu->here() - 0x80;
    sys->db->mem_dump(m0 < 0 ? 0 : m0, 0x80);
#endif // T4_VERBOSE > 1
        
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
            k_vm_exec<<<1, 1, 0, h->st>>>(vm);  // one block per VM
            GPU_CHK();
            cudaEventRecord(h->t1, h->st);
            cudaStreamWaitEvent(h->st, h->t1);  // CPU will wait here
            
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
    k_vm_done<<<1, WARP(VM_COUNT)>>>(vm_pool);
    GPU_CHK();
    for (int i=0; i < VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        GPU_ERR(cudaEventDestroy(h->t1));
        GPU_ERR(cudaEventDestroy(h->t0));
        GPU_ERR(cudaStreamDestroy(h->st));
    }
    MM_FREE(vm_pool);
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
    
    const string APP = string(T4_APP_NAME) + " " + T4_VERSION;
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
    f->setup();
//    f->run();

    cout << APP << " done." << endl;
    f->teardown();

    return 0;
}


    
