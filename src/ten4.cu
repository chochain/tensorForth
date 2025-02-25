/** -*- c++ -*-
 * @file
 * @brief TensorForth class - tensorForth main driver between CUDA and host
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 * Benchmark: 1K*1K cycles on 3.2GHz AMD, Nvidia GTX1660
 *    + 19.0 ms - REALLY SLOW! Probably due to heavy branch divergence.
 *    + 21.1 ms - without NXT cache in nest()            => branch is slow
 *    + 19.1 ms - without push/pop WP                    => static ram access is fast
 *    + 20.3 ms - 16-bit IU, token indirect threading    => not that much worse but portable
 *    + 14.1 ms - 32-bit IU, nest with primitive, indirect threading (with offset)
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
    if (id < T4_VM_COUNT) {
        VM *vm = pool[id].vm = new VM_TYPE(id, *sys);
        vm->init();
    }
    if (id==0) {
        sys->mu->dict_validate();
        sys->mu->status();
        pool[0].vm->state = QUERY;
    }
}

__KERN__ void
k_vm_done(VM_Handle *pool) {
    const auto g  = cg::this_thread_block();   ///< all blocks in grid
    const int  id = g.thread_rank();           ///< VM id
    if (id >= T4_VM_COUNT) return;
    
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

    if (id < T4_VM_COUNT) {
        vm_state s = pool[id].vm->state;
        atomicAdd(&vmst_cnt[s], 1);
    }
}

__KERN__ void
k_vm_exec0(VM *vm) {
    __shared__ DU ss[T4_SS_SZ];      ///< shared mem for ss, rs (much faster)
    __shared__ DU rs[T4_RS_SZ];      ///< shared mem for ss, rs (much faster)

    if (vm->state == STOP) return;
    
    const auto g = cg::this_thread_block();         ///< all blocks
    const int  i = g.thread_rank();                 ///< thread id -> ss[i]
    ///
    /// * enter ForthVM outer loop
    /// * Note: single-threaded, dynamic parallelism when needed
    ///
    if (i == 0) {
        DU *s0 = vm->ss.v;
        DU *r0 = vm->rs.v;
        MEMCPY(ss, s0, sizeof(DU) * T4_SS_SZ);     /// * TODO: parallel sync issue
        MEMCPY(rs, r0, sizeof(DU) * T4_RS_SZ);     /// * see _exec1 below
        vm->ss.v = ss;
        vm->rs.v = rs;
        
        vm->outer();                               /// * enter VM outer loop
        
        MEMCPY(s0, ss, sizeof(DU) * T4_SS_SZ);
        MEMCPY(r0, rs, sizeof(DU) * T4_RS_SZ);
        vm->ss.v = s0;
        vm->rs.v = r0;
    }
}

__KERN__ void
k_vm_exec1(VM *vm) {
    __shared__ DU ss[T4_SS_SZ];      ///< shared mem for ss, rs (much faster)
    __shared__ DU rs[T4_RS_SZ];
    
    if (vm->state == STOP) return;
    
    const auto g = cg::this_thread_block();       ///< all blocks
    const int  i = g.thread_rank();               ///< thread id -> ss[i]

    DU *s0 = vm->ss.v;
    DU *r0 = vm->rs.v;
    
    ///> copy stacks from global to shared mem
    for (int n=0; n < T4_SS_SZ; n+= WARP_SZ)      /// * TODO: parallel copy, sync issue
        if ((n+i) < T4_SS_SZ) ss[n+i] = s0[n+i];  /// * see _exec0 above
    for (int n=0; n < T4_RS_SZ; n+= WARP_SZ)      
        if ((n+i) < T4_RS_SZ) rs[n+i] = r0[n+i];
    g.sync();
    ///
    /// * enter ForthVM outer loop
    /// * Note: single-threaded, dynamic parallelism when needed
    ///
    if (i == 0) {
        vm->ss.v = ss;                            /// * use shared memory ss, rs
        vm->rs.v = rs;
        
        vm->outer();                              /// * enter VM outer loop

        vm->ss.v = s0;                            /// * restore stack pointers
        vm->rs.v = r0;
    }
    g.sync();
    
    ///> copy updated stacks back to global mem
    for (int n=0; n < T4_SS_SZ; n+= WARP_SZ)
        if ((n+i) < T4_SS_SZ) s0[n+i] = ss[n+i];
    for (int n=0; n < T4_RS_SZ; n+= WARP_SZ)
        if ((n+i) < T4_RS_SZ) r0[n+i] = rs[n+i];
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
         << ", vmss["  << T4_SS_SZ << "*" << T4_VM_COUNT << "]"
         << ", vmrs["  << T4_RS_SZ << "*" << T4_VM_COUNT << "]"
         << ", sizeof(Code)="  << sizeof(Code)
         << ", sizeof(Param)=" << sizeof(Param)
         << endl;
    ///
    /// allocate tensorForth system memory blocks
    ///
    sys = System::get_sys(cin, cout, khz, T4_VERBOSE);
    ///
    /// allocate VM handle pool
    ///
    MM_ALLOC(&vm_pool, sizeof(VM_Handle) * T4_VM_COUNT);
    MM_ALLOC(&vmst_cnt, sizeof(int) * 4);
}

__HOST__ void
TensorForth::setup() {
    for (int i=0; i < T4_VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        GPU_ERR(cudaStreamCreate(&h->st));          /// * allocate stream
        GPU_ERR(cudaEventCreate(&h->t0));           /// * allocate timers
        GPU_ERR(cudaEventCreate(&h->t1));
    }
    k_vm_init<<<1, WARP(T4_VM_COUNT)>>>(sys, vm_pool);         /// * initialize all VMs
    GPU_CHK();
}
///
/// collect VM states into vmst_cnt
///
__HOST__ int
TensorForth::more_job() {
    k_ten4_tally<<<1, WARP(T4_VM_COUNT)>>>(vmst_cnt, vm_pool);
    GPU_CHK();
    return vmst_cnt[STOP] < T4_VM_COUNT;          /// * number of STOP VM
}

__HOST__ void
TensorForth::run() {
    ///
    ///> execute VM per stream
    ///
    for (int i=0; i<T4_VM_COUNT; i++) {
        VM_Handle *h  = &vm_pool[i];
        VM        *vm = h->vm;
        
        cudaEventRecord(h->t0, h->st);            /// * record start clock
        k_vm_exec0<<<1, 1, 0, h->st>>>(vm);
        cudaEventRecord(h->t1, h->st);            /// * record end clock
    }
    GPU_CHK();
}

__HOST__ void
TensorForth::profile() {
    int t = sys->trace();
    if (t==0) return;

    INFO("VM.state[STOP,HOLD,QUERY,NEST]=[");
    for (int i = 0; i < 4; i++) INFO(" %d", vmst_cnt[i]);
    INFO(" ]\n");
    if (t > 1) {
        int m0 = (int)sys->mu->here() - 0x80;
        sys->db->mem_dump(m0 < 0 ? 0 : m0, 0x80);
    }
    INFO("VM.dt=[ ");
    for (int i=0; i<T4_VM_COUNT; i++) {
        VM_Handle *h  = &vm_pool[i];
        float dt;
        cudaEventElapsedTime(&dt, h->t0, h->t1);
        INFO("%0.2f ", dt);
    }
    INFO("]\n");
}

__HOST__ int
TensorForth::main_loop() {
//    sys->db->self_tests();
    int i = 0;
    while (more_job() && sys->readline()) {    /// * with loop guard
        if (++i > 200) break;
        run();
        sys->flush();                          /// * flush output buffer
        profile();
//    sys->mu->sweep();       /// * CC: device function call
    }
    return 0;
}

__HOST__ void
TensorForth::teardown(int sig) {
    cout << "\\ VM[] ";
    k_vm_done<<<1, WARP(T4_VM_COUNT)>>>(vm_pool);
    GPU_CHK();
    cout << "freed" << endl;
    
    for (int i=0; i < T4_VM_COUNT; i++) {
        VM_Handle *h = &vm_pool[i];
        GPU_ERR(cudaEventDestroy(h->t1));
        GPU_ERR(cudaEventDestroy(h->t0));
        GPU_ERR(cudaStreamDestroy(h->st));
    }
    MM_FREE(vmst_cnt);           /// * release ten4 Managed memory
    MM_FREE(vm_pool);
    
    System::free_sys();          /// * release system
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
    
    Options opt;
    opt.parse(argc, argv);
    
    GPU_ERR(cudaDeviceSetLimit(cudaLimitStackSize, T4_PER_THREAD_STACK));
    // GPU_ERR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16*1024*1024));
    
    if (opt.help) {
        opt.print_usage(cout);
        opt.check_devices(cout);
        cout << "\nRecommended GPU: " << opt.device_id << endl;
        return 0;
    }
    else opt.check_devices(cout, false);

    cout << T4_APP_NAME << endl;

    TensorForth *f = new TensorForth(opt.device_id, opt.verbose);
    f->setup();
    f->main_loop();
    f->teardown();
    
    cout << T4_APP_NAME << " done." << endl;

    return 0;
}


    
