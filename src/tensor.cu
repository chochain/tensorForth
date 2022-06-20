/**
 * @file
 * @brief tensorForth tensor class implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"
__KERN__ void k_matrix_randomize(U8 *mat, int nrow, int ncol, int seed=0)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < nrow && j < ncol) {
        int off = i + j * nrow;      /* column major */

        // Generate arbitrary elements.
        int const k = 16807;
        int const m = 16;
        DU v = DU(((off + seed) * k % m) - m / 2);

        mat[off] = v;
    }
}

__HOST__
Tensor::Tensor(U16 n, U16 h, U16 w, U16 c) :
    size(n * h * w * c * sizeof(DU)),
    dsize(sizeof(DU)),
    rank(4),
    stride({1,1,1,1}),
    shape({n, h, w, c})
{
    cudaMallocManaged((void**)&data, size);
    GPU_CHK();
    printf("tensor(%d,%d,%d,%d) allocated\n", shape[0], shape[1], shape[2], shape[3]);
}

__HOST__
Tensor::Tensor(U16 h, U16 w) :
    size(h * w * sizeof(DU)),
    dsize(sizeof(DU)),
    rank(2),
    stride({0,1,1,0}),
    shape({0, h, w, 0})
{
    cudaMallocManaged((void**)&data, size);
    GPU_CHK();
    printf("matrix(%d,%d) allocated\n", shape[1], shape[2]);
}

__HOST__ Tensor&
Tensor::fill(U8 v) {
    // Clear the allocation.
    cudaMemset((void*)data, v, size);
    GPU_CHK();
    return *this;
}
__HOST__ Tensor&
Tensor::random(int seed) {
    U16 h = shape[1];
    U16 w = shape[2];
    dim3 block(16, 16);
    dim3 grid(
        (h + block.x - 1) / block.x,     /* column major */
        (w + block.y - 1) / block.y
        );

    k_matrix_randomize<<<grid, block>>>(data, h, w, seed);
    GPU_CHK();
    return *this;
}

__GPU__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &C) {
    // D = alpha * A * B + beta * C;
}
