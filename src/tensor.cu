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
    int M, int N, int K,
    DU alpha, DU beta)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M && j < N) {
        DU2 acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + i * K] * B[j + k * N];
        }
        C[j + i * N] = alpha * acc + beta * C[j + i * N];
    }
}
__KERN__ void k_mat_op(                                    ///< TODO: C
    mat_op op,
    DU *A, DU *B, DU *C,
    int M, int N)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M && j < N) {
        int k = j + i * N;
        switch (op) {
        case ADD: C[k] = A[k] + B[k]; break;
        case SUB: C[k] = A[k] - B[k]; break;
        case MUL: C[k] = A[k] * B[k]; break;               /// * convolution
        case DIV: C[k] = A[k] / B[k]; break;
        }
    }
}
__KERN__ void k_transpose(DU *src, DU *dst, int M, int N) { ///< Note: (src, dst), TODO: CDP
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M && j < N) {
        dst[i + j * M] = src[j + i * N];
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
__KERN__ void k_abs(DU *A, int sz) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < sz) A[k] = fabs(A[k]);
}
__KERN__ void k_identity(DU *A, int M, int N, int C) {
    const DU i01[2][4] = {{ DU0, DU0, DU0, DU0 }, { 1.0, 1.0, 1.0, 1.0 }};
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < M && j < N) {
        memcpy(&A[j + i * N], i01[i==j], sizeof(DU) * C); /// * assume x==y return 0|1
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
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__BOTH__ Tensor&
Tensor::mat(mat_op op, Tensor &A, Tensor &B, Tensor &C) {
    const char *opn[] = { "add", "sub", "mul", "div" };
    U16 m = A.H(), n = A.W();
    WARN("Tensor::mat%s M=%d, N=%d\n", opn[op], m, n);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_mat_op<<<grid, block>>>(op, (DU*)A.data, (DU*)B.data, (DU*)C.data, m, n);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor-scalar addition C = A +- n element-wise (Hadamard)
///
__BOTH__ Tensor&
Tensor::mat(mat_op op, Tensor &A, DU v, Tensor &C) {
    const char *opn[] = { "add", "sub", "mul", "div" };
    U16 m = A.H(), n = A.W();
    WARN("Tensor::mat%s M=%d, N=%d\n", opn[op], m, n);
    DU *da = (DU*)A.data, *dc = (DU*)C.data;
    for (int k = 0; k < A.size; k++) {
        switch (op) {
        case ADD: *dc++ = *da++ + v; break;
        case SUB: *dc++ = *da++ - v; break;
        case MUL: *dc++ = *da++ * v; break;
        case DIV: *dc++ = *da++ / v; break;
        }
    }
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
    U16 m = A.H(), n = A.W();
    WARN("Tensor::transpose A[%d,%d]\n", m, n);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_transpose<<<grid, block>>>((DU*)A.data, (DU*)T.data, m, n);
    cudaDeviceSynchronize();
    return T;
}
///
/// matrix inversion (Gauss-Jordan with Pivot)
/// Note: Gauss-Jordan elimination is expensive O(N^3)
/// TODO: CDP
///
__BOTH__ Tensor&
Tensor::inverse(Tensor &A, Tensor &I) {
    U16 m = A.H(), n = A.W();
    WARN("Tensor::inverse[%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return I; }

    DU *aa = (DU*)A.data;
    DU *ii = (DU*)I.data;
    auto swap_rows = [aa, ii, n](U16 u, U16 z) {
        for (U16 k = 0; k < n; k++) {         ///> TODO: swap entire row
            DU ta = aa[k + z * n], ti = ii[k + z * n];
            aa[k + z * n] = aa[k + u * n]; aa[k + u * n] = ta;
            ii[k + z * n] = ii[k + u * n]; ii[k + u * n] = ti;
        }
    };
    auto find_max = [aa, n](U16 z) {
        int u = z;
        for (U16 i = z + 1; i < n; i++) {    ///> TODO: CDP reduce
            if (ABS(aa[z + i * n]) > ABS(aa[z + u * n])) u = i;
        }
        if (ABS(aa[z + u * n]) < DU_EPS) {
            ERROR("Tensor::inverse sigular!\n");
            return -1;
        }
        return u;
    };
    auto diag = [aa, ii, n](U16 z) {
        DU r0 = aa[z + z * n];
        for (U16 k = 0; k < n; k++) {
            U16 i = k + z * n;
            ii[i] /= r0;
            aa[i] /= r0;
        }};
    auto elim = [aa, ii, n](U16 z) {
        for (U16 i = 0; i < n; i++) {
            DU r1 = aa[z + i * n];
            for (U16 k = 0; i!=z && k < n; k++) {
                ii[k + i * n] -= r1 * ii[k + z * n];
                aa[k + i * n] -= r1 * aa[k + z * n];
            }
        }};
    for (U16 z = 0; z < n; z++) {
        int u = find_max(z);
        if (u < 0) break;
        else if (u != z) {
            swap_rows(u, z);
        }
        diag(z);
        elim(z);
    }
    return I;
}
///
/// LU (preprocessed) matrix inversion
/// TODO: CDP
///
__BOTH__ Tensor&
Tensor::inverse(Tensor &LU) {
    U16 m = LU.H(), n = LU.W();
    DU *aa = (DU*)LU.data;
    auto forward = [aa, n](U16 z) {
        for (U16 y = z + 1; y < n; y++) {
            DU r1 = aa[z + y * n];
            for (U16 k = 0; k < z; k++) {               // columns before
                aa[k + y * n] -= aa[k + z * n] * r1;
            }
            aa[z + y * n] = -r1;                        // current z column
        }};
    auto backward = [aa, n](U16 z) {
        DU r0 = 1.0 / aa[z + z * n];
        aa[z + z * n] = r0;                             // diag
        for (U16 k = z + 1; k < n; k++) {               // current z row
            aa[k + z * n] *= r0;
        }
        for (U16 y = 0; y < z; y++) {                   // factorize rows above
            DU r1 = aa[z + y * n];
            aa[z + y *  n] = -r1 * r0;                  // current z column
            for (U16 k = z + 1; k < n; k++) {           // columns after
                aa[k + y * n] -= aa[k + z * n] * r1;
            }
        }};
    
    if (LU.det() < DU_EPS) return LU;
    
    for (U16 z = 0; z < n - 1; z++)  forward(z);
    for (I16 z = n - 1; z >= 0; z--) backward(z);
    
    return LU;
}
///
/// LU decomposition (no Pivot)
/// Note: A stores both L and U in-place to save space
/// TODO: CDP
///
__BOTH__ Tensor&
Tensor::lu(Tensor &A) {
    U16 m = A.H(), n = A.W();
    WARN("Tensor::lu[%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return A; }

    DU *aa = (DU*)A.data;
    auto elim = [aa, n](U16 z) {
        DU ra = aa[z + z * n];
        if (fabs(ra) < DU_EPS) return;       /// * if 0 skip the row
        for (U16 y = z + 1; y < n; y++) {
            DU r1 = aa[z + y * n] / ra;      /// * substitution
            for (U16 k = z; k < n; k++) {
                aa[k + y * n] -= r1 * aa[k + z * n];
            }
            aa[z + y * n] = r1;              /// L stored in A to save space
        }
    };
	for (U16 z = 0; z < n; z++) {
        elim(z);               /// * eliminate variables in upper triangle
	}
    return A;
}
///
/// PLU methods with permutation vector
/// Note: A stores both L and U in-place to save space, use triu, trul to extract
///       P is permutation vector
/// TODO: CDP
///
__BOTH__ Tensor&
Tensor::plu(Tensor &A, Tensor &P) {
    U16 m = A.H(), n = A.W();
    WARN("Tensor::lu[%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return A; }

    DU *aa = (DU*)A.data;
    DU *vp = (DU*)P.data;
    auto swap_rows = [aa, vp, n](U16 u, U16 z) {
        DU t = vp[z]; vp[z] = vp[u]; vp[u] = t;
        for (U16 k = z; k < n; k++) {         ///> TODO: swap entire row
            t = aa[k + z * n];
            aa[k + z * n] = aa[k + u * n];
            aa[k + u * n] = t;
        }
    };
    auto find_max = [aa, n](U16 z) {
        int u = z;
        for (U16 i = z + 1; i < n; i++) {    ///> TODO: CDP reduce
            if (ABS(aa[z + i * n]) > ABS(aa[z + u * n])) u = i;
        }
        if (ABS(aa[z + u * n]) < DU_EPS) {
            WARN("Tensor::lu sigular!\n");
            return -1;
        }
        return u;
    };
    auto elim = [aa, n](U16 z) {
        DU ra = aa[z + z * n];
        if (fabs(ra) < DU_EPS) return;       /// * if 0 skip the row
        for (U16 y = z + 1; y < n; y++) {
            DU r1 = aa[z + y * n] / ra;      /// * substitution
            for (U16 k = z; k < n; k++) {
                aa[k + y * n] -= r1 * aa[k + z * n];
            }
            aa[z + y * n] = r1;              /// L stored in A to save space
        }
    };
	for (U16 z = 0; z < n; z++) {
        int u = find_max(z);   /// * pivot to reduce rounding error
        if (u < 0) return A;
		if (u != z) { 	       /// * swapping row which has maximum xth column element
            swap_rows(u, z);
        }
        elim(z);               /// * eliminate variables in upper triangle
	}
    return A;
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
/// tensor arithmetics
///
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
///=======================================================================
/// linear algebra methods
///=======================================================================
/// matrix determinant
///
__BOTH__ DU
Tensor::det() {
    U16 m = H(), n = W();
    WARN("Tensor::det[%d,%d]\n", m, n);
    
    DU *d = (DU*)data;
    DU v  = 1.0;
    for (U16 z = 0; z < m; z++) v *= d[z + z * n];
    
    return v;
}
///
/// matrix upper triangle
///
__BOTH__ Tensor&
Tensor::triu() {
    U16 m  = H(), n = W();
    WARN("Tensor::upper[%d,%d]\n", m, n);
    
    DU *d = (DU*)data;
    for (U16 z = 1; z < m; z++) {
        for (U16 k = 0; k < z; k++) {
            d[k + z * n] = DU0;
        }
    }
    cudaDeviceSynchronize();
    return *this;
}
///
/// matrix lower triangle with diag filled with 1
///
__BOTH__ Tensor&
Tensor::tril() {
    U16 m = H(), n = W();
    WARN("Tensor::lower[%d,%d]\n", m, n);
    
    DU *d = (DU*)data;
    for (U16 z = 0; z < m; z++) {
        d[z + z * n] = DU0 + 1.0;
        for (U16 k = z + 1; k < n; k++) {
            d[k + z * n] = DU0;
        }
    }
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
__BOTH__ Tensor&
Tensor::abs() {
    WARN("Tensor#abs\n");
    dim3 block(256), grid((size + block.x -1)/block.x);
    k_abs<<<grid, block>>>((DU*)data, size);
    cudaDeviceSynchronize();
    return *this;
}
///=======================================================================
/// Tensor life-cycle ops
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


