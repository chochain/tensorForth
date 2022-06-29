/** -*- c++ -*-
 * @file
 * @brief - GEMM tester
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>          // cin, cout
#include <signal.h>

//===========================================================================================
//
// CUTLASS includes needed for single-precision GEMM kernel
//
// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"
#include "../src/ten4_types.h"
#include "../src/tensor.h"
#include "../src/opt.h"

typedef DU FP;
typedef cudaError_t (*gemm_op)(Tensor &A, Tensor &B, Tensor &C, FP alpha, FP beta);

void benchmark(gemm_op op, Tensor &A, Tensor &B, Tensor &C, FP alpha, FP beta) {
    cudaEvent_t events[2];
    float       runtime_ms;
    for (auto & event : events) {
        cudaEventCreate(&event);
        GPU_CHK();
    }
    cudaEventRecord(events[0]);

    op(A, B, C, alpha, beta);
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

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(Tensor &A, Tensor &B, Tensor &C, FP alpha, FP beta) {
    int M = C.H(), N = C.W(), K = A.W();
    printf("Cutlass.GEMM M=%d, N=%d, K=%d ", M, N, K);
    using LO   = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<FP, LO, FP, LO, FP, LO>;
    Gemm::Arguments args(
        {M, N, K},               // Gemm Problem dimensions
        {(FP const*)A.data, M},  // Tensor-ref for source matrix A
        {(FP const*)B.data, K},  // Tensor-ref for source matrix B
        {(FP*)C.data, M},        // Tensor-ref for source matrix C
        {(FP*)C.data, M},        // Tensor-ref for destination matrix D (may be different memory than source C matrix)
        {alpha, beta}            // Scalars used in the Epilogue
    );
    // Define a CUTLASS GEMM type
    Gemm cutlass_gemm;
    cutlass::Status status = cutlass_gemm(args);
    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //
    return (status==cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Naive reference GEMM computation.
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
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Reference GEMM computation.
cudaError_t ReferenceGemm(Tensor &A, Tensor &B, Tensor &C, FP alpha, FP beta) {
    int M = A.H(), N = B.W(), K = A.W();
    printf("Ref.GEMM M=%d, N=%d, K=%d", M, N, K);
    dim3 block(16, 16);   /* 256 threads */
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    ReferenceGemm_kernel<<<grid, block>>>(M, N, K, (FP*)A.data, (FP*)B.data, (FP*)C.data, alpha, beta);

    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, FP alpha, FP beta) {
    // Define pointers to matrices in GPU device memory.
    //
    // matrix allocation, fill, random
    //
    Tensor tensor_A(M, K); tensor_A.fill(0.0).random(0);
    Tensor tensor_B(K, N); tensor_B.fill(0.0).random(17);
    //
    // reshape test
    //
    U32 sz = M * N;
    Tensor tensor_C(sz); tensor_C.reshape(M, N).fill(0.0).random(101);
    //
    // reset test
    //
    U8  *ref;
    cudaMallocManaged((void**)&ref, (size_t)sz * sizeof(DU));
    GPU_CHK();
    Tensor tensor_R; tensor_R.reset(ref, sz).reshape(M, N).fill(0).random(101);
    //=============================================================================
    // Launch CUTLASS GEMM.
    //
    benchmark(CutlassSgemmNN, tensor_A, tensor_B, tensor_C, alpha, beta);
    //
    // Lanch reference GEMM
    //
    benchmark(ReferenceGemm, tensor_A, tensor_B, tensor_R, alpha, beta);
    //
    // Verify: copy to host and verify equivalence.
    //
    std::vector<FP> host_c(tensor_C.size, 0);
    std::vector<FP> host_r(tensor_R.size, 0);

    tensor_C.copy_to_host(host_c.data());
    tensor_R.copy_to_host(host_r.data());
    //
    // Test for bit equivalence of results.
    //
    if (host_c != host_r) {
        std::cerr << "results different." << std::endl;
        return cudaErrorUnknown;
    }
    cudaFree(ref);
    return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// usage:  t_tensor <M> <N> <K> <alpha> <beta>
//
#define CUTLASS_OPTS 1

int main(int argc, const char **argv) {
    //
    // Parse the command line to obtain GEMM dimensions and scalar values.
    //
#if CUTLASS_OPTS
    Options opt;
    opt.parse(argc, argv);
    
    if (opt.help) {
        opt.check_devices(std::cout);
        opt.print_usage(std::cout);
        return 0;
    }
    cudaError_t result = TestCutlassGemm(
        opt.problem_size.m(),     // GEMM M dimension
        opt.problem_size.n(),     // GEMM N dimension
        opt.problem_size.k(),     // GEMM K dimension
        opt.alpha.real(),         // alpha
        opt.beta.real()           // beta
        );
#else // CUTLASS_OPTS
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
    // Run the CUTLASS GEMM test.
    //
    cudaError_t result = TestCutlassGemm(
        problem[0],     // GEMM M dimension
        problem[1],     // GEMM N dimension
        problem[2],     // GEMM K dimension
        scalars[0],     // alpha
        scalars[1]      // beta
        );
#endif // CUTLASS_OPTS
    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }
    // Exit.
    return result == cudaSuccess ? 0 : -1;
}
