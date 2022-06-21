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

    if (i < ncol && j < nrow) {
        int off = i + j * ncol;      /* row major */

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
    shape({h, w, n, c})
{
    cudaMallocManaged((void**)&data, size);
    GPU_CHK();
    printf("tensor(%d,%d,%d,%d) allocated\n", shape[2], shape[0], shape[1], shape[3]);
}

__HOST__
Tensor::Tensor(U16 h, U16 w) :
    size(h * w * sizeof(DU)),
    dsize(sizeof(DU)),
    rank(2),
    stride({1,1,0,0}),
    shape({h, w, 0, 0})
{
    cudaMallocManaged((void**)&data, size);
    GPU_CHK();
    printf("matrix(%d,%d) allocated\n", shape[0], shape[1]);
}

__HOST__
Tensor::~Tensor()
{
    if (!data) return;
    cudaFree((void*)data);
    switch (rank) {
    case 2: printf("matrix(%d,%d) freed\n", shape[0], shape[1]); break;
    case 4: printf("tensor(%d,%d,%d,%d) freed\n", shape[2], shape[0], shape[1], shape[3]); break;
    default: printf("~Tensor error: rank=%d\n", rank);
    }
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
    int h = H();
    int w = W();
    dim3 block(16, 16);
    dim3 grid(
        (w + block.x - 1) / block.x,     /* row major */
        (h + block.y - 1) / block.y
        );

    k_matrix_randomize<<<grid, block>>>(data, h, w, seed);
    GPU_CHK();
    return *this;
}

__GPU__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &C) {
    // D = alpha * A * B + beta * C;
}
