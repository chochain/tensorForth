/** -*- c++ -*-
 * @File
 * @brief tensorForth tensor class implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"

#define ABS(d) (fabsf(d))  /**< absolute value */
///
/// GEMM kernel (used CUDA dynamic parallelism)
///     C = alpha * A x B + beta * C
///     where A = MxK, B = KxN, C = MxN
///
__KERN__ void k_gemm(                                        ///< 2D only
    DU *A, DU *B, DU *C,   /* HxK, KxW, HxW */
    int H, int W, int K,
    DU alpha, DU beta)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < H && j < W) {
        DU acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + i * K] * B[j + k * W];
        }
        C[j + i * W] = alpha * acc + beta * C[j + i * W];
    }
}
__KERN__ void k_matadd(                                     ///< TODO: C
    DU *A, DU *B, DU *C,
    int H, int W,
    bool sub)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < H && j < W) {
        int k = j + i * W;
        if (sub) C[k] = A[k] - B[k];
        else     C[k] = A[k] + B[k];
    }
}
__KERN__ void k_transpose(DU *src, DU *dst, int H, int W) { ///< Note: (src, dst), TODO: CDP
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < H && j < W) {
        dst[i + j * H] = src[j + i * W];
    }
}
__KERN__ void k_copy(DU *src, DU *dst, int sz) {           ///< Note: (src, dst)
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < sz) dst[k] = src[k];
}
__KERN__ void k_fill(DU *A, DU v, int sz) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < sz) A[k] = v;
}
__KERN__ void k_scale(DU *A, DU v, int sz) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < sz) A[k] *= v;
}
__KERN__ void k_identity(DU *A, int W, int H, int C) {
    const DU i01[2][4] = {{ DU0, DU0, DU0, DU0 }, { 1.0, 1.0, 1.0, 1.0 }};
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < H && j < W) {
        memcpy(&A[j + i * W], i01[i==j], sizeof(DU) * C); /// * assume x==y return 0|1
    }
}
__KERN__ void k_norm_nodiag(double *A, double *I, int D, int n){   ///< TODO: C
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < D && j < D && j==n && i!=j) {
        I[i*D + j] /= A[i*D + n];
        A[i*D + j] /= A[i*D + n];
    }
}
__KERN__ void k_norm_diag(double *A, double *I, int D, int n) {    ///< TODO: C
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < D && j < D & j==n && i==j) {
        I[i*D + j] /= A[i*D + n];
        A[i*D + j] /= A[i*D + n];
    }
}
__KERN__ void k_gaussjordan(double *A, double *I, int D, int n) {  ///< TODO: C
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < D && j < D && j!=n) {
        I[i*D + j] -= I[n*D + j] * A[i*D + n];
        if (i != n){
            A[i*D + j] -= A[n*D + j] * A[i*D + n];
        }
    }
}
__KERN__ void k_inv_zero(double *A, int D, int n) {               ///< TODO: C
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < D && j < D && j!=n && i==n) A[j*D + i] = DU0;
}
///=======================================================================
/// static methods
///
/// tensor GEMM C' = alpha * A x B + beta * C
///
__BOTH__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &C, DU alpha, DU beta) {
    U16 m = A.H(), n = B.W(), k = A.W();
    WARN("GEMM M=%d, N=%d, K=%d a=%f, b=%f\n", m, n, k, alpha, beta);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_gemm<<<grid, block>>>(
        (DU*)A.data, (DU*)B.data, (DU*)C.data,
        m, n, k,
        alpha, beta);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor addition C = A + B or C = A - B
///
__BOTH__ Tensor&
Tensor::add(Tensor &A, Tensor &B, Tensor &C, bool sub) {
    U16 h = A.H(), w = A.W();
    WARN("Tensor::%s M=%d, N=%d\n", sub ? "sub" : "add", h, w);
    dim3 block(16, 16), grid(
        (h + block.x - 1) / block.x,
        (w + block.y - 1) / block.y
    );
    k_matadd<<<grid, block>>>((DU*)A.data, (DU*)B.data, (DU*)C.data, h, w, sub);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
__BOTH__ Tensor&
Tensor::copy(Tensor &A, Tensor &C) {
    WARN("Tensor::copy size=%d\n", A.size);
    dim3 block(256), grid((A.size + block.x -1) / block.x);
    k_copy<<<grid, block>>>((DU*)A.data, (DU*)C.data, A.size);
    cudaDeviceSynchronize();
    return C;
}
__BOTH__ Tensor&
Tensor::transpose(Tensor &A, Tensor &T) {
    U16 h = A.H(), w = A.W();
    WARN("Tensor::transpose M=%d, N=%d\n", h, w);
    dim3 block(16, 16), grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y
    );
    k_transpose<<<grid, block>>>((DU*)A.data, (DU*)T.data, h, w);
    cudaDeviceSynchronize();
    return T;
}
///
/// matrix inversion
/// Note: Gauss-Jordan elimination is expensive O(N^3)
/// TODO: LU and CDP
///
__BOTH__ Tensor&
Tensor::inverse(Tensor &A, Tensor &I) {
    U16 h = A.H(), w = A.W();
    if (h != w) { ERROR("square matrix?"); return I; }

    WARN("Tensor::inverse[%d,%d]\n", h, w);
    DU *aa = (DU*)A.data;
    DU *ii = (DU*)I.data;
    auto swap_rows = [aa, ii, w](U16 u, U16 z) {
        for (U16 k = 0; k < w; k++) {      // swap entire row
            DU ta = aa[k + z * w], ti = ii[k + z * w];
            aa[k + z * w] = aa[k + u * w];
            ii[k + z * w] = ii[k + u * w];
            aa[k + u * w] = ta;
            ii[k + u * w] = ti;
        }
    };
    auto find_max = [aa, ii, w](U16 z) {
        int u = z;
        for (U16 y = z + 1; y < w; y++) {
            if (aa[z + y * w] > aa[z + u * w]) u = y;
        }
        if (ABS(aa[z + u * w]) < DU_EPS) {
            ERROR("Tensor::inverse sigular!\n");
            return -1;
        }
        return u;
    };
    auto diag = [aa, ii, w](U16 z) {
        DU r0 = aa[z + z * w];
        for (U16 k = 0; k < w; k++) {
            U16 i = k + z * w;
            ii[i] /= r0;
            aa[i] /= r0;
        }};
    auto elim = [aa, ii, w](U16 z) {
        for (U16 y = 0; y < w; y++) {
            DU r1 = aa[z + y * w];
            for (U16 k = 0; y!=z && k < w; k++) {
                ii[k + y * w] -= r1 * ii[k + z * w];
                aa[k + y * w] -= r1 * aa[k + z * w];
            }
        }};
    for (U16 z = 0; z < w; z++) {
        int u = find_max(z);
        if (u < 0) break;
        else if (u != z) swap_rows(u, z);
        diag(z);
        elim(z);
    }
    return I;
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
    WARN("tensor[%d] allocated\n", size);
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
    WARN("matrix(%d,%d) allocated\n", shape[0], shape[1]);
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
    WARN("tensor(%d,%d,%d,%d) allocated\n", shape[3], shape[0], shape[1], shape[2]);
}

__HOST__
Tensor::~Tensor()
{
    if (!data) return;
    cudaFree((void*)data);
    switch (rank) {
    case 2: WARN("matrix(%d,%d) freed\n", shape[0], shape[1]); break;
    case 4: WARN("tensor(%d,%d,%d,%d) freed\n", shape[3], shape[0], shape[1], shape[2]); break;
    default: WARN("~Tensor error: rank=%d\n", rank);
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
    WARN("Tensor::reset(%p, %d)\n", mptr, sz);
    dsize  = sizeof(DU);
    size   = sz;
    rank   = 1;
    U16 t[4] = {1, 1, 1, 1};      memcpy(stride, t, sizeof(t));
    U16 s[4] = {(U16)sz,1, 1, 1}; memcpy(shape,  s, sizeof(s));
    attr   = 0;
    data   = (U8*)mptr;
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U32 sz) {
    if (sz == size) {
        reset(data, size);
        WARN("Tensor::reshaped(%d)\n", size);
    }
    else {
        ERROR("Tensor::reshape sz != size (%d != %d)\n", sz, size);
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
        WARN("Tensor::reshaped(%d,%d)\n", shape[0], shape[1]);
    }
    else {
        ERROR("Tensor::reshape sz != size (%d != %d)\n", sz, size);
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
        WARN("Tensor::reshaped(%d,%d,%d,%d)\n", shape[3], shape[0], shape[1], shape[2]);
    }
    else {
        ERROR("Tensor::reshape sz != size (%d != %d)\n", sz, size);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::identity() {
    if (rank < 2) return *this;
    int h = H(), w = W();
    dim3 block(16, 16), grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y
    );
    k_identity<<<grid, block>>>((DU*)data, h, w, C());
    cudaDeviceSynchronize();
    return *this;
}

__BOTH__ Tensor&
Tensor::fill(DU v) {
    WARN("Tensor#fill with %f\n", v);
    dim3 block(256), grid((size + block.x -1)/block.x);
    k_fill<<<grid, block>>>((DU*)data, v, size);
    cudaDeviceSynchronize();
    return *this;
}

__BOTH__ Tensor&
Tensor::scale(DU v) {
    WARN("Tensor#scale by %f\n", v);
    dim3 block(256), grid((size + block.x -1)/block.x);
    k_scale<<<grid, block>>>((DU*)data, v, size);
    cudaDeviceSynchronize();
    return *this;
}

__BOTH__ DU
Tensor::sum() {
    DU v = DU0;
    for (int i=0; i < size; i++) v += ((DU*)data)[i];   ///> TODO: CDP prefix sum
    cudaDeviceSynchronize();
    return v;
}

__BOTH__ DU
Tensor::dot(Tensor &B) {
    DU  acc = DU0;
    if (rank == 1 && B.rank == 1 && size == B.size) {
        for (int k=0; k < size; k++) {                   ///> TODO: kernel
            acc += ((DU*)data)[k] * ((DU*)B.data)[k];
        }
    }
    else ERROR("A.dot(B) dim? %d != %d)\n", size, B.size);
    return acc;
}
