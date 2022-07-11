/** -*- c++ -*-
 * @File
 * @brief tensorForth tensor class implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"
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
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < W && y < H) {
        DU acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + y * K] * B[x + k * W];
        }
        C[x + y * W] = alpha * acc + beta * C[x + y * W];
    }
}
__KERN__ void k_matadd(                                     ///< TODO: C
    DU *A, DU *B, DU *C,
    int H, int W,
    bool sub)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < W && y < H) {
        int i = x + y * W;
        if (sub) C[i] = A[i] - B[i];
        else     C[i] = A[i] + B[i];
    }
}
__KERN__ void k_transpose(DU *dst, DU *src, int H, int W) { ///< TODO: C
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < W && y < H) {
        dst[y + x * H] = src[x + y * W];
    }
}
__KERN__ void k_copy(DU *dst, DU *src, int sz) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < sz) dst[x] = src[x];
}
__KERN__ void k_fill(DU *A, DU v, int sz) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < sz) A[x] = v;
}
__KERN__ void k_scale(DU *A, DU v, int sz) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x < sz) A[x] *= v;
}
__KERN__ void k_identity(DU *A, int W, int H, int C) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < W && y < H) {
        for (int c=0; c < C; c++) A[x + y*W + c] = (x==y) ? 1.0 : DU0;
    }
}
__KERN__ void k_norm_nodiag(double *A, double *I, int D, int n){   ///< TODO: C
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < D && y < D && x==n && x!=y) {
        I[x*D + y] /= A[x*D + n];
        A[y*D + y] /= A[x*D + n];
    }
}
__KERN__ void k_norm_diag(double *A, double *I, int D, int n) {    ///< TODO: C
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < D && y < D & x==n && x==y) {
        I[x*D + y] /= A[x*D + n];
        A[x*D + y] /= A[x*D + n];
    }
}
__KERN__ void k_gaussjordan(double *A, double *I, int D, int n) {  ///< TODO: C
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < D && y < D && x!=n) {
        I[x*D + y] -= I[n*D + y] * A[x*D + n];
        if (y != n){
            A[x*D + y] -= A[n*D + y] * A[x*D + n];
		}
	}
}
__KERN__ void k_inv_zero(double *A, int D, int n) {               ///< TODO: C
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < D && y < D && x!=n && y==n) A[x*D + y] = DU0;
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
Tensor::copy(Tensor &D, Tensor &S) {
    WARN("Tensor::copy size=%d\n", size);
    dim3 block(256), grid((S.size + block.x -1) / block.x);
    k_copy<<<grid, block>>>((DU*)D.data, (DU*)S.data, S.size);
    cudaDeviceSynchronize();
    return D;
}

__BOTH__ Tensor&
Tensor::transpose(Tensor &D, Tensor &S) {
    U16 h = S.H(), w = S.W();
    WARN("Tensor::transpose M=%d, N=%d\n", h, w);
    dim3 block(16, 16), grid(
        (w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y
    );
    k_transpose<<<grid, block>>>((DU*)D.data, (DU*)S.data, h, w);
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
    dsize  = sizeof(DU);
    size   = sz;
    rank   = 1;
    U16 t[4] = {1, 1, 1, 1};      memcpy(stride, t, sizeof(t));
    U16 s[4] = {(U16)sz,1, 1, 1}; memcpy(shape,  s, sizeof(s));
    attr   = 0;
    data   = (U8*)mptr;
    WARN("tensor reset(%p, %d)\n", mptr, sz);
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U32 sz) {
    if (sz == size) {
        reset(data, size);
        WARN("tensor reshaped(%d)\n", size);
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
        WARN("tensor reshaped(%d,%d)\n", shape[0], shape[1]);
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
        WARN("tensor reshaped(%d,%d,%d,%d)\n", shape[3], shape[0], shape[1], shape[2]);
    }
    else {
        ERROR("reshape sz != size (%d != %d)\n", sz, size);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::identity() {
    if (rank < 2) return *this;
    DU *d = (DU*)data;
    for (int j=0; j < H(); j++) {
        for (int i=0; i < W(); i++) {
            for (int c=0; c < C(); c++) *d++ = (i==j) ? 1.0 : DU0;
        }
    }
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


