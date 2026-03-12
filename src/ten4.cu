/** -*- c++ -*-
 * @file
 * @brief TensorForth class - tensorForth main driver between CUDA and host
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 * Benchmark: 1K*1K cycles on 3.2GHz AMD, Nvidia GTX1660
 *    + 9.3ms - ceforth50x as reference
 *    + 19.0s - 2000x slower! Most likely due to heavy branch divergence.
 *    + 21.1s - without NXT cache in nest()         => branch is slow
 *    + 19.1s - without push/pop WP                 => static ram access is fast
 *    + 20.3s - 16-bit IU, token indirect threading => not that much worse but portable
 *    + 11.3s - CUDA 11.6, 32-bit IU, nest with primitive, indirect threading (with offset)
 *    +  7.5s - CUDA 12.6, same code as above
 *    +  5.0s - CUDA 11.4, rollback to Ubuntu 20.04 + 470 driver, from inside docker
 */
#include <iostream>          // cin, cout
#include <signal.h>
#include "ten4.h"            // wrapper

namespace t4 {
using t4::vm::VM;
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
        pool[0].vm->state = vm::QUERY;
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
k_ten4_tally(System *sys, int *vmst_cnt, VM_Handle *pool) {
    const auto g  = cg::this_thread_block();   ///< all blocks in grid
    const int  id = g.thread_rank();           ///< VM id

    if (id==0) sys->mu->sweep();               ///< clear marked free tensors
    g.sync();
    
    if (id < vm::VM_STATE_SZ) vmst_cnt[id] = 0;
    g.sync();

    if (id < T4_VM_COUNT) {
        vm::vm_state s = pool[id].vm->state;
        atomicAdd(&vmst_cnt[s], 1);
    }
}

__KERN__ void
k_vm_exec0(VM *vm) {
    __shared__ DU ss[T4_SS_SZ];      ///< shared mem for ss, rs (much faster)
    __shared__ DU rs[T4_RS_SZ];      ///< shared mem for ss, rs (much faster)

    if (vm->state == vm::STOP) return;
    
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
        
        if (vm->state==vm::HOLD) vm->resume();     /// * resume holding work
        else                     vm->outer();      /// * enter VM outer loop
        
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

    if (vm->state == vm::STOP) return;
    
    const auto g = cg::this_thread_block();       ///< all blocks
    const int  i = g.thread_rank();               ///< thread id -> ss[i]
    auto cpy = [i](DU *dst, DU *src) {
        for (int n=i; n < T4_SS_SZ; n+= WARP_SZ)
            if (n < T4_SS_SZ) dst[n] = src[n];
    };
    DU *s0 = vm->ss.v;
    DU *r0 = vm->rs.v;
    
    ///> copy stacks from global to shared mem
    cpy(ss, s0); cpy(rs, r0); g.sync();
    ///
    /// * enter ForthVM outer loop
    /// * Note: single-threaded, dynamic parallelism when needed
    ///
    if (i == 0) {
        vm->ss.v = ss;                            /// * use shared memory ss, rs
        vm->rs.v = rs;

        if (vm->state==vm::HOLD) vm->resume();    /// * resume holding work
        else                     vm->outer();     /// * enter VM outer loop

        vm->ss.v = s0;                            /// * restore stack pointers
        vm->rs.v = r0;
    }
    g.sync(); cpy(s0, ss); cpy(r0, rs);           ///> copy updated stacks back to global mem
}

TensorForth::TensorForth(int device, int verbose) {
    ///
    /// set active device
    ///
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "\nERR: failed to activate GPU " << device << "\n";
        exit(1);
    }
    ///
    /// query GPU shader clock rate
    ///
    int khz = 0;
    GPU_ERR(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device));

    std::cout << "\\ GPU "  << device
              << " at "     << khz/1000 << "MHz"
              << ", dict["  << T4_DICT_SZ << "]"
              << ", pmem="  << T4_PMEM_SZ/1024 << "K"
              << ", ostor=" << T4_OSTORE_SZ/1024/1024 << "M"
              << ", vmss["  << T4_SS_SZ << "*" << T4_VM_COUNT << "]"
              << ", vmrs["  << T4_RS_SZ << "*" << T4_VM_COUNT << "]"
              << std::endl;
    ///
    /// allocate tensorForth system memory blocks
    ///
    sys = System::get_sys(std::cin, std::cout, khz, T4_VERBOSE);
    ///
    /// allocate VM handle pool
    ///
    MM_ALLOC(&vm_pool, sizeof(VM_Handle) * T4_VM_COUNT);
    MM_ALLOC(&vmst_cnt, sizeof(int) * vm::VM_STATE_SZ);   /// * 4 states + 1 VM[0].hold
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
    k_ten4_tally<<<1, WARP(T4_VM_COUNT)>>>(sys, vmst_cnt, vm_pool);
    GPU_CHK();
    return vmst_cnt[vm::STOP] < T4_VM_COUNT;      /// * number of STOP VM
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
        
        k_vm_exec0<<<1, 1, 0, h->st>>>(vm);       /// * each VM on their own stream
        
        cudaEventRecord(h->t1, h->st);            /// * record end clock
        cudaStreamSynchronize(h->st);             /// * ensure managed mem updated
    }
    GPU_CHK();
}

__HOST__ void
TensorForth::profile() {
    int t = sys->trace();
    if (t==0) return;

    DEBUG("VM.state[STOP,HOLD,QUERY,NEST]=[");
    for (int i = 0; i < 4; i++) DEBUG(" %d", vmst_cnt[i]);
    DEBUG(" ]\n");
    if (t > 1) {
        int m0 = (int)sys->mu->here() - 0x80;
        sys->db->mem_dump(m0 < 0 ? 0 : m0, 0x80);
    }
    TRACE("VM.dt=[ ");
    for (int i=0; i<T4_VM_COUNT; i++) {
        VM_Handle *h  = &vm_pool[i];
        float dt;
        cudaEventElapsedTime(&dt, h->t0, h->t1);
        TRACE("%0.2f ", dt);
    }
    TRACE("]\n");
}

__HOST__ int
TensorForth::main_loop() {
    // sys->db->self_tests();
    int i = 0;
    while (more_job() && sys->readline(vmst_cnt[vm::HOLD])) {
//        if (++i > 200) break;                  /// * runaway loop guard TODO: CC
        run();
        sys->flush();                          /// * flush output buffer
        profile();
    }
    return 0;
}

__HOST__ void
TensorForth::teardown(int sig) {
    std::cout << "\\ VM[] ";
    k_vm_done<<<1, WARP(T4_VM_COUNT)>>>(vm_pool);
    GPU_CHK();
    std::cout << "freed" << std::endl;
    
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

} // namespace t4
///
/// main program
///
void sigsegv_handler(int sig, siginfo_t *si, void *arg) {
    std::cout << "Exception caught at: " << si->si_addr << std::endl;
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
        opt.print_usage(std::cout);
        opt.check_devices(std::cout);
        std::cout << "\nRecommended GPU: " << opt.device_id << std::endl;
        return 0;
    }
    else opt.check_devices(std::cout, false);

    std::cout << T4_APP_NAME << std::endl;

    t4::TensorForth *f = new t4::TensorForth(opt.device_id, opt.verbose);
    f->setup();
    f->main_loop();
    f->teardown();

    std::cout << T4_APP_NAME << " done." << std::endl;

    return 0;
}
    
