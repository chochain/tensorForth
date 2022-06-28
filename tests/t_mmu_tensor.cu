/** -*- c++ -*-
 * @file
 * @brief - tensorForth mmu#tensor tests
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>          // cin, cout
#include <signal.h>
#include <sstream>
using namespace std;

#include "../src/ten4_types.h"
#include "../src/mmu.h"
//
// GEMM kernel (used CUDA dynamic parallelism)
//     C = alpha * A x B + beta * C
//     where A = MxK, B = KxN, C = MxN
//
__KERN__ void k_GEMM_test(
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
// GEMM test driver kernel code
//
__KERN__ void
test_mmu_gemm(
    MMU *mmu, int khz,
    U16 M, U16 N, U16 K, DU alpha, DU beta) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Tensor &A = mmu->tensor(M, K); A.fill(0).random(0);
    Tensor &B = mmu->tensor(K, N); B.fill(0).random(17);
    Tensor &C = mmu->tensor(M, N); C.fill(0).random(101);

    int m = C.H(), n = C.W(), k = A.W();
    printf("\nGEMM M=%d, N=%d, K=%d", m, n, k);
    
    dim3 block(16, 16);         /* 256 threads */
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    clock_t t0 = clock();
    k_GEMM_test<<<grid, block>>>(
        m, n, k,
        (DU*)A.data, (DU*)B.data, (DU*)C.data,
        alpha, beta);
    cudaDeviceSynchronize();     // deprecated in 11.6! What is the alternative?
    printf(" (k_GEMM in %0.2f ms @ %d khz)", ((float)(clock() - t0))/khz, khz);
}

cudaError_t benchmark(MMU *mmu, int khz, U16 M, U16 N, U16 K, DU alpha, DU beta) {
    cudaEvent_t events[2];
    float       runtime_ms;
    for (auto & event : events) {
        cudaEventCreate(&event);
        GPU_CHK();
    }
    cudaEventRecord(events[0]);

    test_mmu_gemm<<<1,1>>>(mmu, khz, M, N, K, alpha, beta);
    cudaError_t error = cudaGetLastError();

    // Wait for work on the device to complete.
    cudaEventRecord(events[1]);
    cudaEventSynchronize(events[1]);
    // Measure eresultd runtime
    cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    printf(" => %0.2f ms\n", runtime_ms);

    // Cleanup
    for (auto event : events) {
        (void)cudaEventDestroy(event);
    }
    return error;
}
///
/// usage:  t_mmu_tensor <M> <N> <K> <alpha> <beta>
///
int main(int argc, const char *arg[]) {
    //
    // Parse the command line to obtain GEMM dimensions and scalar values.
    //
    // GEMM problem dimensions.
    int problem[3] = { 1024, 512, 2048 };

    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(arg[i]);
        ss >> problem[i - 1];
    }
    // Scalars used for linear scaling the result of the matrix product.
    float scalars[2] = { 1, 0 };
    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(arg[i]);
        ss >> scalars[i - 4];
    }
    //
    // Run the MMU-driven GEMM test.
    //
    int device = 0;
    int khz    = 0;
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device);
    GPU_CHK();
    
    MMU *mmu   = new MMU();
    cudaError_t result = benchmark(
        mmu,
        khz, 
        problem[0],     // GEMM M dimension
        problem[1],     // GEMM N dimension
        problem[2],     // GEMM K dimension
        scalars[0],     // alpha
        scalars[1]      // beta
        );

    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }
    // Exit.
    return result == cudaSuccess ? 0 : -1;
}
