/** -*- c++ -*-
 * @File
 * @brief tensorForth tensor class implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"
///
/// kernel methods (static) CDP-ready
///
#define CDP(g) \
    int i = threadIdx.x + blockIdx.x * blockDim.x;  \
    int j = threadIdx.y + blockIdx.y * blockDim.y;  \
    if (i < N && j < M) { g; }
///
/// GEMM kernel (used CUDA dynamic parallelism)
///     C = alpha * A x B + beta * C
///     where A = MxK, B = KxN, C = MxN
///
__KERN__ void k_gemm(
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
__KERN__ void k_matadd(
    int M, int N,
    DU *A, DU *B, DU *C,
    bool sub)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < N && j < M) {
        int off = i + j * N;
        if (sub) C[off] = A[off] - B[off];
        else     C[off] = A[off] + B[off];
    }
}

__KERN__ void k_copy(DU *dst, DU *src, int nrow, int ncol) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < ncol && j < nrow) {
        int off = i + j * ncol;
        dst[off] = src[off];
    }
}
__KERN__ void k_transpose(DU *dst, DU *src, int nrow, int ncol) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < ncol && j < nrow) {
        dst[j + i * nrow] = src[i + j * ncol];
    }
}
__KERN__ void k_scale(DU *A, DU v, int nrow, int ncol) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < ncol && j < nrow) {
        A[i + j * ncol] *= v;
    }
}
///=======================================================================
/// static methods
///
/// tensor GEMM C' = alpha * A x B + beta * C
///
__BOTH__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &C, DU alpha, DU beta) {
    U16 m = A.H(), n = B.W(), k = A.W();
    DEBUG("GEMM M=%d, N=%d, K=%d a=%f, b=%f\n", m, n, k, alpha, beta);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_gemm<<<grid, block>>>(
        m, n, k,
        (DU*)A.data, (DU*)B.data, (DU*)C.data,
        alpha, beta);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor addition C = A + B or C = A - B
///
__BOTH__ Tensor&
Tensor::add(Tensor &A, Tensor &B, Tensor &C, bool sub) {
    U16 m = A.H(), n = A.W();
    DEBUG("Tensor::%s M=%d, N=%d\n", sub ? "sub" : "add", m, n);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_matadd<<<grid, block>>>(m, n, (DU*)A.data, (DU*)B.data, (DU*)C.data, sub);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
    
__BOTH__ Tensor&
Tensor::copy(Tensor &D, Tensor &S) {
    U16 m = S.H(), n = S.W();
    DEBUG("Tensor::copy M=%d, N=%d\n", m, n);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_copy<<<grid, block>>>((DU*)D.data, (DU*)S.data, m, n);
    cudaDeviceSynchronize();
    return D;
}

__BOTH__ Tensor&
Tensor::transpose(Tensor &D, Tensor &S) {
    U16 m = S.H(), n = S.W();
    DEBUG("Tensor::transpose M=%d, N=%d\n", m, n);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_transpose<<<grid, block>>>((DU*)D.data, (DU*)S.data, m, n);
    cudaDeviceSynchronize();
    return D;
}
///=======================================================================
/// Tensor class constructors
///
__HOST__
Tensor::Tensor() :
    dsize(sizeof(DU)),
    size(0),
    rank(0),
    stride{1, 1, 1, 1},
    shape{1, 1, 1, 1} {}

__HOST__
Tensor::Tensor(U32 sz) :
    dsize(sizeof(DU)),
    size(sz),
    rank(1),
    stride{1, 1, 1, 1},
    shape{(U16)sz, 1, 1, 1} {
    cudaMallocManaged((void**)&data, (size_t)size * dsize);
    GPU_CHK();
    DEBUG("tensor[%d] allocated\n", size);
}

__HOST__
Tensor::Tensor(U16 h, U16 w) :
    dsize(sizeof(DU)),
    size(h * w),
    rank(2),
    stride{1, 1, 1, 1},
    shape{h, w, 1, 1} {
    cudaMallocManaged((void**)&data, (size_t)size * dsize);
    GPU_CHK();
    DEBUG("matrix(%d,%d) allocated\n", shape[0], shape[1]);
}

__HOST__
Tensor::Tensor(U16 n, U16 h, U16 w, U16 c) :
    dsize(sizeof(DU)),
    size(n * h * w * c),
    rank(4),
    stride{1, 1, 1, 1},
    shape{h, w, n, c} {
    cudaMallocManaged((void**)&data, (size_t)size * dsize);
    GPU_CHK();
    DEBUG("tensor(%d,%d,%d,%d) allocated\n", shape[3], shape[0], shape[1], shape[2]);
}

__HOST__
Tensor::~Tensor()
{
    if (!data) return;
    cudaFree((void*)data);
    switch (rank) {
    case 2: DEBUG("matrix(%d,%d) freed\n", shape[0], shape[1]); break;
    case 4: DEBUG("tensor(%d,%d,%d,%d) freed\n", shape[3], shape[0], shape[1], shape[2]); break;
    default: DEBUG("~Tensor error: rank=%d\n", rank);
    }
}
///=======================================================================
/// Tensor class methods
///
__BOTH__ Tensor&
Tensor::set_as_view(bool set) {
    if (set) attr |= T4_TENSOR_VIEW;
    else     attr &= ~T4_TENSOR_VIEW;
    return *this;
}

__BOTH__ Tensor&
Tensor::reset(void *mptr, U32 sz) {
    dsize  = sizeof(DU);
    size   = sz;
    rank   = 1;
    U16 t[4] = {1, 1, 1, 1};      memcpy(stride, t, sizeof(t));
    U16 s[4] = {(U16)sz,1, 1, 1}; memcpy(shape,  s, sizeof(s));
    attr   = 0;
    data   = (U8*)mptr;
    DEBUG("tensor reset(%p, %d)\n", mptr, sz);
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U32 sz) {
    if (sz == size) {
        reset(data, size);
        DEBUG("tensor reshaped(%d)\n", size);
    }
    else {
        ERROR("reshape sz != size (%d != %d)\n", sz, size);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U16 h, U16 w) {
    U32 sz = h * w;
    if (sz == size) {
        rank   = 2;
        U16 t[4] = {1, 1, 1, 1}; memcpy(stride, t, sizeof(t));
        U16 s[4] = {h, w, 1, 1}; memcpy(shape,  s, sizeof(s));
        DEBUG("tensor reshaped(%d,%d)\n", shape[0], shape[1]);
    }
    else {
        ERROR("reshape sz != size (%d != %d)\n", sz, size);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U16 n, U16 h, U16 w, U16 c) {
    U32 sz = n * h * w * c;
    if (sz == size) {
        rank   = 4;
        U16 t[4] = {1, 1, 1, 1}; memcpy(stride, t, sizeof(t));
        U16 s[4] = {h, w, c, n}; memcpy(shape,  s, sizeof(s));
        DEBUG("tensor reshaped(%d,%d,%d,%d)\n", shape[3], shape[0], shape[1], shape[2]);
    }
    else {
        ERROR("reshape sz != size (%d != %d)\n", sz, size);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::fill(DU v) {
    DU  *d = (DU*)data;
    for (int i=0; i<size; i++) *d++ = v;
    return *this;
}

__BOTH__ Tensor&
Tensor::scale(DU v) {
    int h = H(), w = W();
    DEBUG("Tensor#scale by %f\n", v);
    dim3 block(16, 16), grid(
        (w + block.x - 1) / block.x,     /* row major */
        (h + block.y - 1) / block.y
        );
    k_scale<<<grid, block>>>((DU*)data, v, h, w);
    return *this;
}

__BOTH__ DU
Tensor::sum() {
    DU v  = DU0;
    DU *d = (DU*)data;
    for (int i=0; i<size; i++) v += *d++;  /// * TODO: CDP prefix sum
    return v;
}

__BOTH__ DU
Tensor::dot(Tensor &B) {
    DU  acc = DU0;
    if (rank == 1 && B.rank == 1 && size == B.size) {
        for (int k=0; k < size; k++) {
            acc += ((DU*)data)[k] * ((DU*)B.data)[k];
        }
    }
    else ERROR("A.dot(B) dim? %d != %d)\n", size, B.size);
    return acc;
}


