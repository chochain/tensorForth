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
    t4_mat_op op,
    DU *A, DU *B, DU *C,
    int M, int N)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M && j < N) {
        int k = j + i * N;
        switch (op) {                                      /// no divergence
        case ADD: C[k] = A[k] + B[k]; break;
        case SUB: C[k] = A[k] - B[k]; break;
        case MUL: C[k] = A[k] * B[k]; break;               /// * convolution
        case DIV: C[k] = A[k] / B[k]; break;
        }
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
        A.data, B.data, C.data,
        m, n, k,
        alpha, beta);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__BOTH__ Tensor&
Tensor::mat(t4_mat_op op, Tensor &A, Tensor &B, Tensor &C) {
    const char *opn[] = { "add", "sub", "mul", "div" };
    U16 m = A.H(), n = A.W();
    WARN("Tensor::mat%s M=%d, N=%d\n", opn[op], m, n);
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_mat_op<<<grid, block>>>(op, A.data, B.data, C.data, m, n);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor-scalar addition C = A +- n element-wise (Hadamard)
///
__BOTH__ Tensor&
Tensor::mat(t4_mat_op op, Tensor &A, DU v, Tensor &C) {
    const char *opn[] = { "add", "sub", "mul", "div" };
    U16 m = A.H(), n = A.W();
    WARN("Tensor::mat%s M=%d, N=%d\n", opn[op], m, n);
    DU *dc = C.data, *da = A.data;
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
    k_copy<<<grid, block>>>(A.data, C.data, A.size);
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
    k_transpose<<<grid, block>>>(A.data, T.data, m, n);
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
    DU *da = A.data, *di = I.data;
    auto swap_rows = [da, di, n](U16 u, U16 z) {
        for (U16 k = 0; k < n; k++) {         ///> TODO: swap entire row
            DU ta = da[k + z * n], ti = di[k + z * n];
            da[k + z * n] = da[k + u * n]; da[k + u * n] = ta;
            di[k + z * n] = di[k + u * n]; di[k + u * n] = ti;
        }
    };
    auto find_max = [da, n](U16 z) {
        int u = z;
        for (U16 i = z + 1; i < n; i++) {    ///> TODO: CDP reduce
            if (ABS(da[z + i * n]) > ABS(da[z + u * n])) u = i;
        }
        if (ABS(da[z + u * n]) < DU_EPS) {
            ERROR("Tensor::inverse sigular!\n");
            return -1;
        }
        return u;
    };
    auto diag = [da, di, n](U16 z) {
        DU r0 = da[z + z * n];
        for (U16 k = 0; k < n; k++) {
            U16 i = k + z * n;
            di[i] /= r0;
            da[i] /= r0;
        }};
    auto elim = [da, di, n](U16 z) {
        for (U16 i = 0; i < n; i++) {
            DU r1 = da[z + i * n];
            for (U16 k = 0; i!=z && k < n; k++) {
                di[k + i * n] -= r1 * di[k + z * n];
                da[k + i * n] -= r1 * da[k + z * n];
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
    DU *dd = LU.data;
    auto forward = [dd, n](U16 z) {
        for (U16 y = z + 1; y < n; y++) {
            DU r1 = dd[z + y * n];
            for (U16 k = 0; k < z; k++) {               // columns before
                dd[k + y * n] -= dd[k + z * n] * r1;
            }
            dd[z + y * n] = -r1;                        // current z column
        }};
    auto backward = [dd, n](U16 z) {
        DU r0 = DU1 / dd[z + z * n];
        dd[z + z * n] = r0;                             // diag
        for (U16 k = z + 1; k < n; k++) {               // current z row
            dd[k + z * n] *= r0;
        }
        for (U16 y = 0; y < z; y++) {                   // factorize rows above
            DU r1 = dd[z + y * n];
            dd[z + y *  n] = -r1 * r0;                  // current z column
            for (U16 k = z + 1; k < n; k++) {           // columns after
                dd[k + y * n] -= dd[k + z * n] * r1;
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

    DU *da = A.data;
    auto elim = [da, n](U16 z) {
        DU ra = da[z + z * n];
        if (fabs(ra) < DU_EPS) return;      /// * if 0 skip the row
        for (U16 y = z + 1; y < n; y++) {
            DU r1 = da[z + y * n] / ra;     /// * substitution
            for (U16 k = z; k < n; k++) {
                da[k + y * n] -= r1 * da[k + z * n];
            }
            da[z + y * n] = r1;             /// L stored in A to save space
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

    DU *da = A.data, *dp = P.data;
    auto swap_rows = [da, dp, n](U16 u, U16 z) {
        DU t = dp[z]; dp[z] = dp[u]; dp[u] = t;
        for (U16 k = z; k < n; k++) {         ///> TODO: swap entire row
            t = da[k + z * n];
            da[k + z * n] = da[k + u * n];
            da[k + u * n] = t;
        }
    };
    auto find_max = [da, n](U16 z) {
        int u = z;
        for (U16 i = z + 1; i < n; i++) {    ///> TODO: CDP reduce
            if (ABS(da[z + i * n]) > ABS(da[z + u * n])) u = i;
        }
        if (ABS(da[z + u * n]) < DU_EPS) {
            WARN("Tensor::lu sigular!\n");
            return -1;
        }
        return u;
    };
    auto elim = [da, n](U16 z) {
        DU ra = da[z + z * n];
        if (fabs(ra) < DU_EPS) return;       /// * if 0 skip the row
        for (U16 y = z + 1; y < n; y++) {
            DU r1 = da[z + y * n] / ra;      /// * substitution
            for (U16 k = z; k < n; k++) {
                da[k + y * n] -= r1 * da[k + z * n];
            }
            da[z + y * n] = r1;              /// L stored in A to save space
        }
    };
    for (U16 z = 0; z < n; z++) {
        int u = find_max(z);   /// * pivot to reduce rounding error
        if (u < 0) return A;
        if (u != z) {          /// * swapping row which has maximum xth column element
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
    for (int i=0; i < size; i++) v += data[i];   ///> TODO: CDP prefix sum
    cudaDeviceSynchronize();
    return v;
}

__BOTH__ DU
Tensor::dot(Tensor &B) {
    DU  acc = DU0;
    if (rank == 1 && B.rank == 1 && size == B.size) {
        for (int k=0; k < size; k++) {                   ///> TODO: kernel
            acc += data[k] * B.data[k];
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

    DU v  = DU1;
    for (U16 z = 0; z < m; z++) v *= data[z + z * n];

    return v;
}
///
/// matrix upper triangle
///
__BOTH__ Tensor&
Tensor::triu() {
    U16 m  = H(), n = W();
    WARN("Tensor::upper[%d,%d]\n", m, n);

    for (U16 z = 1; z < m; z++) {
        for (U16 k = 0; k < z; k++) {
            data[k + z * n] = DU0;
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

    for (U16 z = 0; z < m; z++) {
        data[z + z * n] = DU1;
        for (U16 k = z + 1; k < n; k++) {
            data[k + z * n] = DU0;
        }
    }
    cudaDeviceSynchronize();
    return *this;
}
__BOTH__ Tensor&
Tensor::map(t4_mat_op op, DU v) {
    const char *opn[] = { "+", "-", "*", "/", "abs", "exp", "tanh", "relu", "fill", "scale" };
    WARN("Tensor#%s v=%f\n", opn[op], v);
    dim3 block(256), grid((size + block.x -1)/block.x);
    switch(op) {
    case ABS:   k_abs<<<  grid, block>>>(data, size);    break;
    case EXP:   k_exp<<<  grid, block>>>(data, size);    break;
    case TANH:  k_tanh<<< grid, block>>>(data, size);    break;
    case RELU:  k_relu<<< grid, block>>>(data, size);    break;
    case FILL:  k_fill<<< grid, block>>>(data, v, size); break;
    case SCALE: k_scale<<<grid, block>>>(data, v, size); break;
    default: ERROR("Tensor#map op=%d?\n", op); break;
    }
    cudaDeviceSynchronize();
    return *this;
}
///=======================================================================
/// Tensor life-cycle ops
///
__BOTH__ Tensor&
Tensor::set_as_view(bool set) {
    ttype = set ? VIEW : TENSOR;
    return *this;
}
__BOTH__ Tensor&
Tensor::reset(void *mptr, U32 sz) {
    WARN("Tensor::reset(%p, %d)\n", mptr, sz);
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { (U16)sz, 1, 1, 1 };
    const DU  g[4] = { DU0, DU0, DU0, DU0 };
    size    = sz;
    dsize   = sizeof(DU);
    rank    = 1;
    ttype   = TENSOR;
    data    = (DU*)mptr;
    grad_fn = NONE;
    memcpy(stride, s, sizeof(s));
    memcpy(shape,  t, sizeof(t));
    memcpy(grad,   g, sizeof(g));
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
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { h, w, 1,  1};
    U32 sz = h * w;
    if (sz == size) {
        rank   = 2;
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        WARN("Tensor::reshaped(%d,%d)\n", shape[0], shape[1]);
    }
    else {
        ERROR("Tensor::reshape sz != size (%d != %d)\n", sz, size);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U16 n, U16 h, U16 w, U16 c) {
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { h, w, c, n };
    U32 sz = n * h * w * c;
    if (sz == size) {
        rank   = 4;
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
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
    int m = H(), n = W();
    dim3 block(16, 16), grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    k_identity<<<grid, block>>>(data, m, n, sizeof(DU)*C());
    cudaDeviceSynchronize();
    return *this;
}
