/*!
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

#include "ten4_config.h"
#include "aio.h"             // CUDA async IO
#include "eforth.h"          // eForth core
#include "ten4.h"            // wrapper

#define MAJOR_VERSION        "1"
#define MINOR_VERSION        "0"

__GPU__ ForthVM *vm_pool[VM_MIN_COUNT];
///
/// instantiate VMs (threadIdx.x is vm_id)
///
__KERN__ void
ten4_init(int khz, Istream *istr, Ostream *ostr, MMU *mmu) {
    int i = threadIdx.x;
    if (i >= VM_MIN_COUNT) return;

    ForthVM *vm = vm_pool[i] = new ForthVM(khz, istr, ostr, mmu);  // instantiate VM
    vm->ss.init(mmu->vss(i), T4_SS_SZ);  // point data stack to managed memory block

    if (i==0) vm->init();                // initialize common dictionary (once only)
}
///
/// check VM status (using parallel reduction - overkill?)
///
__KERN__ void
ten4_busy(int *busy) {
    extern __shared__ bool b[];          // share memory for fast calc

    int i = threadIdx.x;
    b[i] = (i < VM_MIN_COUNT) ? vm_pool[i]->status==VM_RUN : 0;
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
/// tensorForth kernel - VM dispatcher
///
#include <stdio.h>
__KERN__ void
ten4_exec() {
    const char *st[] = {"READY", "RUN", "WAITING", "STOPPED"};
    extern __shared__ DU shared_ss[];
    if (threadIdx.x!=0) return;

    int      b   = blockIdx.x;
    ForthVM *vm  = vm_pool[b];
    DU      *ss  = &shared_ss[b * T4_SS_SZ];    // adjust stack pointer based on VM id
    DU      *ss0 = vm->ss.v;                    // capture VM data stack
    MEMCPY(ss, ss0, sizeof(DU) * T4_SS_SZ);     // copy stack into shared memory block
    vm->ss.v = ss;                              // redirect data stack to shared memory

    if (vm->status == VM_RUN) vm->outer();
    else printf("VM[%d] %s\n", blockIdx.x, st[vm->status]);

    __syncthreads();
    MEMCPY(ss0, ss, sizeof(DU) * T4_SS_SZ);     // copy updated stack to managed memory
    vm->ss.v = ss0;                             // restore stack back to VM
}

TensorForth::TensorForth(int device, bool trace) {
    ///
    /// set active device
    ///
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "\nERR: failed to activate GPU %d\n", device);
        exit(1);
    }
    ///
    /// query GPU shader clock rate
    ///
    int khz = 0;
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device);
    GPU_CHK();
    ///
    /// allocate cuda memory blocks
    ///
    mmu = new MMU();                            ///> instantiate memory manager
    aio = new AIO(mmu, trace);                  ///> instantiate async IO manager
    cudaMalloc((void**)&busy, sizeof(int));     ///> allocate managed busy flag
    GPU_CHK();
    ///
    /// instantiate virtual machines
    ///
    int t = WARP(VM_MIN_COUNT);                 ///> thread count = 32 modulo
    ten4_init<<<1, t>>>(khz, aio->istream(), aio->ostream(), mmu); // create VMs
    GPU_CHK();

#if T4_VERBOSE
    cout << "GPU " << device
         << " initialized at " << khz/1000 << "MHz"
         << ", dict["          << T4_DICT_SZ << "]"
         << ", pmem="          << T4_PMEM_SZ/1024 << "K"
         << ", tensor="        << T4_TENSOR_SZ/1024/1024 << "M"
#if CC_DEBUG
         << ", sizeof(Code)=" << sizeof(Code)
#endif // CC_DEBUG
         << endl;
#endif // T4_VERBOSE
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
    ten4_busy<<<1, t, t * sizeof(bool)>>>(busy);
    GPU_SYNC();
    //UNLOCK();               // TODO:

    cudaMemcpy(&h_busy, busy, sizeof(int), D2H);

    return h_busy;
}

#define VSS_SZ (sizeof(DU)*T4_SS_SZ*VM_MIN_COUNT)
__HOST__ int
TensorForth::run() {
    while (is_running()) {
        if (aio->readline()) {        // feed from host console to managed input buffer
            ten4_exec<<<1, 1, VSS_SZ>>>();
            GPU_CHK();
            aio->flush();             // flush output buffer
        }
        yield();
#if MMU_DEBUG
        int m0 = (int)mmu->here() - 0x80;
        mmu->mem_dump(cout, m0 < 0 ? 0 : m0, 0x80);
#endif // MMU_DEBUG
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

int main0(int argc, char**argv) {
    string app = string(T4_APP_NAME) + " " + MAJOR_VERSION + "." + MINOR_VERSION;
    sigtrap();
    TensorForth *f = new TensorForth();

    cout << app << endl;
    f->run();

    cout << app << " done." << endl;
    f->teardown();

    return 0;
}

typedef DU FP;
typedef cudaError_t (*gemm_op)(Tensor &A, Tensor &B, Tensor &C, FP alpha, FP beta);

__KERN__ void ReferenceGemm_kernel(
    int M, int N, int K,
    FP *A, FP *B, FP *C,   /* MxK, KxN, MxN */
    FP  alpha, FP beta)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < N && j < M) {
        FP acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + j * K] * B[i + k * N];      /* row major */
        }
        C[i + j * N] = alpha * acc + beta * C[i + j * N];
    }
}

__KERN__ void
test_mmu_gemm(MMU *mmu, U16 M, U16 N, U16 K, FP alpha, FP beta) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Tensor &A = mmu->tensor(M, K);
    Tensor &B = mmu->tensor(K, N);
    Tensor &C = mmu->tensor(M, N);

    int m = C.H(), n = C.W(), k = A.W();
    printf("\nRef.GEMM M=%d, N=%d, K=%d", m, n, k);
    
    dim3 block(16, 16);   /* 256 threads */
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    
    ReferenceGemm_kernel<<<grid, block>>>(m, n, k, (FP*)A.data, (FP*)B.data, (FP*)C.data, alpha, beta);
}

void benchmark(MMU *mmu) {
    cudaEvent_t events[2];
    float       runtime_ms;
    for (auto & event : events) {
        cudaEventCreate(&event);
        GPU_CHK();
    }
    cudaEventRecord(events[0]);

    test_mmu_gemm<<<1,1>>>(mmu, 1024, 512, 2048, 1.5, 1.5);
    GPU_CHK();

    // Wait for work on the device to complete.
    cudaEventRecord(events[1]);
    cudaEventSynchronize(events[1]);
    // Measure eresultd runtime
    cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    printf(" => %f ms\n", runtime_ms);

    // Cleanup
    for (auto event : events) {
        (void)cudaEventDestroy(event);
    }
}

int main(int argc, char**argv) {
    MMU *mmu = new MMU();

    benchmark(mmu);

    return 0;
}

    
