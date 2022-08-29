/** -*- c++ -*-
 * @File
 * @brief tensorForth tensor class implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"

#define ABS(d) (fabsf(d))  /**< absolute value */
///=======================================================================
/// static methods
///
/// matrix-matrix multiplication with transpose, increment options
///
__KERN__ void
k_ten_op(t4_ten_op op, float *t, int sz, float v=DU0) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < sz) {
        switch(op) {
        case O_FILL:  t[k] = v;                         break;
        case O_SCALE: t[k] *= v;                        break;
        case O_POW:   t[k] = POW(t[k], v);              break;
        case O_ABS:   t[k] = ABS(t[k]);                 break;
        case O_EXP:   t[k] = EXP(t[k]);                 break;
        case O_LOG:   t[k] = LOG(t[k]);                 break;
        case O_TANH:  t[k] = TANH(t[k]);                break;
        case O_RELU:  t[k] = t[k] > DU0 ? t[k] : DU0;   break;
        case O_SIGM:  t[k] = DU1 / (DU1 + EXP(-t[k]));  break;
        }
    }
}
__KERN__ void
k_matmul(
    DU *A, DU *B, DU *C,   /* C[MxN] = A[MxK] @ B[KxN] */
    int M, int N, int K,
    t4_mm_opt opt)
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = j + i * N;

    if (i < M && j < N) {                                  /// * TODO: tiled
        DU2 acc = DU0;
        if (opt & MM_A_TXP) {                              /// * no divergence
            for (int k = 0; k < K; ++k) {
                acc += A[i + k * M] * B[j + k * N];        /// * transpose A
            }
        }
        else if (opt & MM_B_TXP) {                         /// * transpose B
            for (int k = 0; k < K; ++k) {
                acc += A[k + i * K] * B[k + j * K];
            }
        }
        else {
            for (int k = 0; k < K; ++k) {
                acc += A[k + i * K] * B[j + k * N];
            }
        }
        if (opt & MM_INC) C[z] += acc;                     /// * increment C
        else              C[z] =  acc;
    }
}
///
/// GEMM kernel (used CUDA dynamic parallelism)
///     C = alpha * A x B + beta * C
///     where A = MxK, B = KxN, C = MxN
///
__KERN__ void
k_gemm(                       ///< 2D only, TODO: C
    DU *A, DU *B, DU *C,      /* C[MxN] = a * A[MxK] @ B[KxN] + b * C[MxN] */
    int M, int N, int K,
    DU alpha, DU beta)
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = j + i * N;

    if (i < M && j < N) {                                  /// * TODO: tiled
        DU2 acc = DU0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + i * K] * B[j + k * N];
        }
        C[z] = alpha * acc + beta * C[z];                  /// * scaling
    }
}
///
/// matrix-matrix element-wise ops
///
__KERN__ void k_mat_op(                                    ///< 2D only, TODO: C
    t4_ten_op op,
    DU *A, DU *B, DU *C,
    int M, int N)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < M && j < N) {
        int k = j + i * N;
        switch (op) {                                      /// no divergence
        case O_ADD: C[k] = A[k] + B[k]; break;
        case O_SUB: C[k] = A[k] - B[k]; break;
        case O_MUL: C[k] = A[k] * B[k]; break;             /// * convolution
        case O_DIV: C[k] = A[k] / B[k]; break;
        }
    }
}
__BOTH__ Tensor&
Tensor::mm(
    Tensor &A, Tensor &B, Tensor &C, t4_mm_opt opt) {
    U16 M  = opt & MM_A_TXP ? A.W() : A.H();
    U16 Ka = opt & MM_A_TXP ? A.H() : A.W();
    U16 N  = opt & MM_B_TXP ? B.H() : B.W();
    U16 Kb = opt & MM_B_TXP ? B.W() : B.H();
    if (Ka != Kb) {
        ERROR("Tensor#mm Ka(%d)!=Kb(%d)\n", Ka, Kb);
        return C;
    }
    WARN("Tensor#matmul M=%d, N=%d, K=%d\n", M, N, Ka);
    printf("Tensor#matmul M=%d, N=%d, K=%d\n", M, N, Ka);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ), grd(
        (N + blk.x - 1) / blk.x,
        (M + blk.y - 1) / blk.y
    );
    k_matmul<<<grd,blk>>>(A.data, B.data, C.data, M, N, Ka, opt);
    cudaDeviceSynchronize();
    return C;
}
/// tensor GEMM C' = alpha * A x B + beta * C
///
__BOTH__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &C, DU alpha, DU beta) {
    U16 m = A.H(), n = B.W(), ka = A.W(), kb = B.H();
    if (ka != kb) {
        ERROR("Tensor#gemm ka(%d)!=kb(%d)\n", ka, kb);
        return C;
    }
    WARN("GEMM M=%d, N=%d, K=%d a=%f, b=%f\n", m, n, ka, alpha, beta);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ), grd(
        (n + blk.x - 1) / blk.x,
        (m + blk.y - 1) / blk.y
    );
    ///
    /// TODO: cudaLaunchKernel is host mode only (as of CUDA 11.6)
    ///
    k_gemm<<<grd, blk>>>(A.data, B.data, C.data, m, n, ka, alpha, beta);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__BOTH__ Tensor&
Tensor::mat(t4_ten_op op, Tensor &A, Tensor &B, Tensor &C) {
    U16 m = A.H(), n = A.W();
    OPN("add", "sub", "mul", "div");
    WARN("Tensor::mat%s M=%d, N=%d\n", opn[op], m, n);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ), grd(
        (n + blk.x - 1) / blk.x,
        (m + blk.y - 1) / blk.y
    );
    k_mat_op<<<grd, blk>>>(op, A.data, B.data, C.data, m, n);
    cudaDeviceSynchronize();     // TODO: deprecated 11.6, use cooperative_groups.sync()
    return C;
}
///
/// tensor-scalar addition C = A +- n element-wise (Hadamard)
///
__BOTH__ Tensor&
Tensor::mat(t4_ten_op op, Tensor &A, DU v, Tensor &C) {
    U16 m = A.H(), n = A.W();
    OPN("add", "sub", "mul", "div");
    WARN("Tensor::mat%s M=%d, N=%d\n", opn[op], m, n);
    DU *dc = C.data, *da = A.data;
    for (int k = 0; k < A.numel; k++) {
        switch (op) {
        case O_ADD: *dc++ = *da++ + v; break;
        case O_SUB: *dc++ = *da++ - v; break;
        case O_MUL: *dc++ = *da++ * v; break;
        case O_DIV: *dc++ = *da++ / v; break;
        }
    }
    return C;
}
__BOTH__ Tensor&
Tensor::copy(Tensor &A, Tensor &C) {
    WARN("Tensor::copy numel=%d\n", A.numel);
    dim3 blk(T4_WARP_SZ * T4_WARP_SZ), grd((A.numel + blk.x -1) / blk.x);
    k_copy<<<grd, blk>>>(A.data, C.data, A.numel);
    cudaDeviceSynchronize();
    return C;
}
__BOTH__ Tensor&
Tensor::transpose(Tensor &A, Tensor &T) {
    U16 m = A.H(), n = A.W();
    WARN("Tensor::transpose A[%d,%d]\n", m, n);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ), grd(
        (n + blk.x - 1) / blk.x,
        (m + blk.y - 1) / blk.y
    );
    k_transpose<<<grd, blk>>>(A.data, T.data, m, n);
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
/// LU (preprocessed) matrix inversion
/// TODO: CDP
///
__BOTH__ Tensor&
Tensor::lu_inverse(Tensor &LU) {
    U16 m = LU.H(), n = LU.W();
    DU *dd = LU.data;
    auto forward = [dd, n](int z) {
        for (int y = z + 1; y < n; y++) {
            DU r1 = dd[z + y * n];
            for (int k = 0; k < z; k++) {               // columns before
                dd[k + y * n] -= dd[k + z * n] * r1;
            }
            dd[z + y * n] = -r1;                        // current z column
        }};
    auto backward = [dd, n](int z) {
        DU r0 = DU1 / dd[z + z * n];
        dd[z + z * n] = r0;                             // diag
        for (int k = z + 1; k < n; k++) {               // current z row
            dd[k + z * n] *= r0;
        }
        for (int y = 0; y < z; y++) {                   // factorize rows above
            DU r1 = dd[z + y * n];
            dd[z + y *  n] = -r1 * r0;                  // current z column
            for (int k = z + 1; k < n; k++) {           // columns after
                dd[k + y * n] -= dd[k + z * n] * r1;
            }
        }};
    
    if (LU.det() < DU_EPS) return LU;
    
    for (int z = 0; z < n - 1; z++)  forward(z);
    for (int z = n - 1; z >= 0; z--) backward(z);
    
    return LU;
}
///
/// PLU methods with permutation vector
/// Note: A stores both L and U in-place to save space, use triu, trul to extract
///       P is permutation vector
/// TODO: CDP
///
__BOTH__ Tensor&
Tensor::plu(Tensor &A, Tensor &P, int *ns) {
    U16 m = A.H(), n = A.W();
    WARN("Tensor::plu[%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return A; }

    DU *da = A.data, *dp = P.data;
    *ns = 0;                                  ///> initialize flip sign
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
    for (U16 z = 0; z < m; z++) dp[z] = z;   /// init permutation vector
    for (U16 z = 0; z < n; z++) {
        int u = find_max(z);   /// * pivot to reduce rounding error
        if (u < 0) return A;
        if (u != z) {          /// * swapping row which has maximum xth column element
            swap_rows(u, z);
            *ns += 1;
        }
        elim(z);               /// * eliminate variables in upper triangle
    }
    return A;
}
///=======================================================================
/// tensor arithmetics
///
__BOTH__ DU
Tensor::sum() {
    DU v = DU0;
    for (int i=0; i < numel; i++) v += data[i];  ///> TODO: CDP prefix sum
    cudaDeviceSynchronize();
    return SCALAR(v);
}
__BOTH__ DU
Tensor::avg() {
    DU v = sum() / numel;
    return SCALAR(v);
}
__BOTH__ DU
Tensor::std() {
    DU sum = DU0, avg = this->avg();
    DU *d  = data;
    for (int i=0; i < numel; i++, d++) {
        DU v = *d - avg;
        sum += v * v;
    }
    return numel ? sqrtf(sum / numel) : DU0;
}
__BOTH__ DU
Tensor::max() {
    DU v = data[0];
    for (int i=1; i < numel; i++) {              ///> TODO: CDP prefix sum
        v = MAX(data[i], v);
    }
    cudaDeviceSynchronize();
    return SCALAR(v);
}
__BOTH__ DU
Tensor::min() {
    DU v = data[0];
    for (int i=1; i < numel; i++) {              ///> TODO: CDP prefix sum
        v = MIN(data[i], v);
    }
    cudaDeviceSynchronize();
    return SCALAR(v);
}
__BOTH__ DU
Tensor::dot(Tensor &B) {
    DU  acc = DU0;
    if (rank == 1 && B.rank == 1 && numel == B.numel) {
        for (int k=0; k < numel; k++) {          ///> TODO: kernel
            acc += data[k] * B.data[k];
        }
    }
    else ERROR("A.dot(B) dim? %d != %d)\n", numel, B.numel);
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

    DU v = DU1;
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
Tensor::map(t4_ten_op op, DU v) {
    OPN("+", "-", "*", "/", "@", "x", "fill", "scale","pow", "abs", "exp", "log", "tanh", "relu", "sigmoid");
    WARN("Tensor#%s v=%f\n", opn[op], v);
    dim3 blk(T4_WARP_SZ*T4_WARP_SZ), grd((numel + blk.x -1)/blk.x);
    k_ten_op<<<grd, blk>>>(op, data, numel, v);
    cudaDeviceSynchronize();
    return *this;
}
///=======================================================================
/// Tensor life-cycle ops
///
__BOTH__ Tensor&
Tensor::reset(void *mptr, U32 sz, t4_obj tt, t4_layer fn) {
    WARN("Tensor::reset(%p, %d)\n", mptr, sz);
    const U16 s[4] = { 1, 1, 1, 1 };
    const U16 t[4] = { (U16)sz, 1, 1, 1 };
    const DU  g[4] = { DU0, DU0, DU0, DU0 };
    numel   = sz;
    dsize   = DSIZE;
    rank    = 1;
    ttype   = tt;
    data    = (DU*)mptr;
    grad_fn = fn;
    memcpy(stride, s, sizeof(s));
    memcpy(shape,  t, sizeof(t));
    memcpy(grad,   g, sizeof(g));
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U32 sz) {
    if (sz == numel) {
        reset(data, numel, (t4_obj)ttype, grad_fn);   /// preserve ttype and fn
        WARN("Tensor::reshaped(%d)\n", numel);
    }
    else {
        ERROR("Tensor::reshape sz != numel (%d != %d)\n", sz, numel);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U16 h, U16 w) {
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { h, w, 1,  1};
    U32 sz = h * w;
    if (sz == numel) {
        rank   = 2;
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        WARN("Tensor::reshaped(%d,%d)\n", shape[0], shape[1]);
    }
    else {
        ERROR("Tensor::reshape sz != numel (%d != %d)\n", sz, numel);
    }
    return *this;
}

__BOTH__ Tensor&
Tensor::reshape(U16 n, U16 h, U16 w, U16 c) {
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { h, w, c, n };
    U32 sz = n * h * w * c;
    if (sz == numel) {
        rank   = 4;
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        WARN("Tensor::reshaped(%d,%d,%d,%d)\n", shape[3], shape[0], shape[1], shape[2]);
    }
    else {
        ERROR("Tensor::reshape sz != numel (%d != %d)\n", sz, numel);
    }
    return *this;
}
__BOTH__ Tensor&
Tensor::reshape(U16 c1, U16 n, U16 h, U16 w, U16 c) {
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { h, w, c, n };
    U32 sz = c1 * n * h * w * c;
    if (sz == numel) {
        rank = 5;
        parm = c1;        /// use parm field, so we don't need s[5]
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        WARN("Tensor::reshaped(%d,%d,%d,%d,%d)\n", c1, shape[3], shape[0], shape[1], shape[2]);
    }
    else {
        ERROR("Tensor::reshape sz != numel (%d != %d)\n", sz, numel);
    }
    return *this;
}
__BOTH__ Tensor&
Tensor::identity() {
    if (rank < 2) return *this;
    int M = H(), N = W();
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ), grd(
        (N + blk.x - 1) / blk.x,
        (M + blk.y - 1) / blk.y
    );
    k_identity<<<grd, blk>>>(data, M, N, C()*sizeof(DU));
    cudaDeviceSynchronize();
    return *this;
}
