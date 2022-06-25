/*!
 * @file - ten4.cu
 * @brief - tensorForth mmu#tensor tests
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>          // cin, cout
#include <signal.h>
using namespace std;

#include "../src/ten4_config.h"
#include "../src/ten4_types.h"
#include "../src/mmu.h"

__KERN__ void ReferenceGemm_kernel(
    int M, int N, int K,
    DU *A, DU *B, DU *C,   /* MxK, KxN, MxN */
    DU alpha, DU beta)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < N && j < M) {
        DU acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + j * K] * B[i + k * N];      /* row major */
        }
        C[i + j * N] = alpha * acc + beta * C[i + j * N];
    }
}
//
// dynamic parallel
//
__KERN__ void
test_mmu_gemm(MMU *mmu, U16 M, U16 N, U16 K, DU alpha, DU beta) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Tensor &A = mmu->tensor(M, K);
    Tensor &B = mmu->tensor(K, N);
    Tensor &C = mmu->tensor(M, N);

    int m = C.H(), n = C.W(), k = A.W();
    printf("\nRef.GEMM M=%d, N=%d, K=%d", m, n, k);
    
    dim3 block(16, 16);         /* 256 threads */
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    ReferenceGemm_kernel<<<grid, block>>>(
        m, n, k,
        (DU*)A.data, (DU*)B.data, (DU*)C.data,
        alpha, beta);
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

    