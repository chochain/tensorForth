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
__KERN__ void
k_vm_init(System *sys, VM_Handle *pool) {
    const auto g  = cg::this_thread_block();   ///< all blocks in grid
    const int  id = g.thread_rank();           ///< VM id
    if (id < VM_COUNT) {
        VM *vm = pool[id].vm = new VM_TYPE(id, sys);
        vm->init();
    }
    if (id==0) {
        sys->mu->status();
        pool[0].vm->state = QUERY;
    }
}

__KERN__ void
k_vm_done(VM_Handle *pool) {
    const auto g  = cg::this_thread_block();   ///< all blocks in grid
    const int  id = g.thread_rank();           ///< VM id
    if (id >= VM_COUNT) return;
    
    delete pool[id].vm;
}
///
/// check VM status (using warp-level collectives)
///

__KERN__ void
k_ten4_tally(int *vmst_cnt, VM_Handle *pool) {
    const auto g  = cg::this_thread_block();   ///< all blocks in grid
    const int  id = g.thread_rank();           ///< VM id

    if (id < 4) vmst_cnt[id] = 0;
    g.sync();

    if (id < VM_COUNT) {
        vm_state s = pool[id].vm->state;
        atomicAdd(&vmst_cnt[s], 1);
    }
}

__KERN__ void
k_vm_exec(VM *vm) {
    __shared__ DU ss[T4_SS_SZ];      ///< shared mem for ss, rs (much faster)
    __shared__ DU rs[T4_RS_SZ];

    if (vm->state==STOP) return;
    
    const auto g = cg::this_thread_block();  ///< all blocks
    const int  i = g.thread_rank();          ///< thread id -> ss[i]
    DU *s0 = vm->ss.v;
    DU *r0 = vm->rs.v;

    ///> copy stacks from global to shared mem
    for (int n=0; n < T4_SS_SZ; n+= WARP_SZ)
        if ((n+i) < vm->ss.idx) ss[n+i] = s0[n+i];
    for (int n=0; n < T4_RS_SZ; n+= WARP_SZ)
        if ((n+i) < vm->rs.idx) rs[n+i] = r0[n+i];
    g.sync();
    
    if (i == 0) {
        vm->ss.v = ss;
        vm->rs.v = rs;
        ///
        /// * enter ForthVM outer loop
        /// * Note: single-threaded, dynamic parallelism when needed
        ///
        vm->outer();                               /// * enter VM outer loop
    }
    g.sync();
    
    ///> copy updated stacks back to global mem
    for (int n=0; n < T4_SS_SZ; n+= WARP_SZ)
        if ((n+i) < vm->ss.idx) s0[n+i] = ss[n+i];
    for (int n=0; n < T4_RS_SZ; n+= WARP_SZ)
        if ((n+i) < vm->rs.idx) r0[n+i] = rs[n+i];

    if (i == 0) {
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
    MM_ALLOC(&vmst_cnt, sizeof(int) * 4);
}

__HOST__ void
TensorForth::setup() {
    for (int i=0; i < VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        GPU_ERR(cudaStreamCreate(&h->st));          /// * allocate stream
        GPU_ERR(cudaEventCreate(&h->t0));           /// * allocate timers
        GPU_ERR(cudaEventCreate(&h->t1));
    }
    k_vm_init<<<1, WARP(VM_COUNT)>>>(sys, vm_pool);         /// * initialize all VMs
    GPU_CHK();
}

__HOST__ int
TensorForth::tally() {
    if (sys->trace() <= 1) return 0;

    k_ten4_tally<<<1, WARP(VM_COUNT)>>>(vmst_cnt, vm_pool);
    GPU_CHK();
    
    cout << "VM.state[STOP,HOLD,QUERY,NEST]=[";
    for (int i = 0; i < 4; i++) cout << " " << vmst_cnt[i];
    cout << " ]" << endl;

#if T4_VERBOSE > 1
    int m0 = (int)sys->mu->here() - 0x80;
    sys->db->mem_dump(m0 < 0 ? 0 : m0, 0x80);
#endif // T4_VERBOSE > 1
    
    return vmst_cnt[STOP];                         /// * number of STOP VM
}

__HOST__ void
TensorForth::run() {
    ///
    ///> execute VM per stream
    ///
    for (int i=0; i<VM_COUNT; i++) {
        VM_Handle *h  = &vm_pool[i];
        VM        *vm = h->vm;
        
        cudaEventRecord(h->t0, h->st);            /// * record start clock
        k_vm_exec<<<1, 1, 0, h->st>>>(vm);
        cudaEventRecord(h->t1, h->st);            /// * record end clock
    }
    GPU_CHK();
    ///
    ///> profile
    ///
    for (int i=0; i<VM_COUNT; i++) {
        VM_Handle *h  = &vm_pool[i];
        float dt;
        cudaEventElapsedTime(&dt, h->t0, h->t1);
        TRACE("VM[%d] dt=%0.3f\n", i, dt);
    }
}

__HOST__ int
TensorForth::main_loop() {
    while (tally() < VM_COUNT && sys->readline()) {
        run();
        sys->flush();              /// * flush output buffer
//    sys->mu->sweep();       /// * CC: device function call
    }    
    return 0;
}

__HOST__ void
TensorForth::teardown(int sig) {
    cout << "\\ VM[] ";
    k_vm_done<<<1, WARP(VM_COUNT)>>>(vm_pool);
    GPU_CHK();
    cout << "freed" << endl;
    
    for (int i=0; i < VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        GPU_ERR(cudaEventDestroy(h->t1));
        GPU_ERR(cudaEventDestroy(h->t0));
        GPU_ERR(cudaStreamDestroy(h->st));
    }
    MM_FREE(vmst_cnt);           /// * release ten4 Managed memory
    MM_FREE(vm_pool);
    delete sys;                  /// * release system
    
    cudaDeviceReset();
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
    f->main_loop();
    f->teardown();
    
    cout << APP << " done." << endl;

    return 0;
}


    
