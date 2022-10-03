/** -*- c++ -*-
 * @file
 * @brief Tensor class - ranked tensor impmementation i.e. vector, matrix, tensor, ...
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
    const int k = threadIdx.x + blockIdx.x * blockDim.x;
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
///
/// array sum
/// Note: tiled_partition<32> used
///
__KERN__ void
k_sum(DU *A, DU *sum, int N) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    DU vj = j < N ? A[j] : DU0;
    ///
    /// prefix sum every 32-threaded tile
    ///
    auto t = cg::tiled_partition<32>(cg::this_thread_block());
    auto shfl_sum = [](cg::thread_block_tile<32> t, DU v) {
        for (int k = 16; k > 0; k >>= 1) {
            v += t.shfl_down(v, k);
        }
        return v;
    };
    DU tt = shfl_sum(t, vj);
    ///
    /// sum up atomic
    ///
    if (t.thread_rank() == 0) atomicAdd(sum, tt);
}
__KERN__ void
k_matmul(
    DU *A, DU *B, DU *O,   /* O[MxN] = A[MxK] @ B[KxN] */
    int M, int N, int K,
    t4_mm_opt opt)
{
    const int tx = threadIdx.x, i = tx + blockIdx.y * blockDim.y;
    const int ty = threadIdx.y, j = ty + blockIdx.x * blockDim.x;
    const int c  = blockIdx.z, C = gridDim.z;               ///> channels
    const int z  = c + (j + i * N) * C;

    if (i < M && j < N && c < C) {                         /// * TODO: tiled
        DU  *ax, *bx;
        int ai, bi;
        if (opt & MM_A_TXP) {                              /// * no divergence
            ax = &A[c + i * C];     ai = M * C;
            bx = &B[c + j * C];     bi = N * C;
        }
        else if (opt & MM_B_TXP) {                         /// * transpose B
            ax = &A[c + i * K * C]; ai = C;
            bx = &B[c + j * K * C]; bi = C;
        }
        else {
            ax = &A[c + i * K * C]; ai = C;
            bx = &B[c + j * C];     bi = N * C;
        }
        DU2 acc = DU0;
//      acc += ax[k * C] * bx[k * N * C];                  /// * 8.1 ms 1Kx1K
        for (int k = 0; k < K; k++, ax += ai, bx += bi) {
            acc += (*ax) * (*bx);                          /// * 6.2 ms 1Kx1K
        }
        if (opt & MM_INC) O[z] += acc;                     /// * increment O
        else              O[z] =  acc;
    }
}
///
/// GEMM kernel (used CUDA dynamic parallelism)
///     O = alpha * A x B + beta * O
///     where A = MxK, B = KxN, O = MxN
///
__KERN__ void
k_gemm(
    DU *A, DU *B, DU *O,      /* O[MxN] = a * A[MxK] @ B[KxN] + b * O[MxN] */
    int M, int N, int K,
    DU alpha, DU beta)
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int c = blockIdx.z, C = gridDim.z;               ///> channels
    const int NC= N * C;
    const int z = c + j * C + i * NC;

    if (i < M && j < N && c < C) {                         /// * TODO: tiled
        DU2 acc = DU0;
        DU *ax = &A[c + i * K * C];
        DU *bx = &B[c + j * C];
        for (int k = 0; k < K; k++, ax += C, bx += NC) {
            acc += (*ax) * (*bx);
        }
        O[z] = alpha * acc + beta * O[z];                  /// * scaling
    }
}
///
/// matrix-matrix element-wise ops
///
__KERN__ void k_mat_op(
    t4_ten_op op,
    DU *A, DU *B, DU *O,
    int M, int N)
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int c = blockIdx.z, C = gridDim.z;               ///> channels

    if (i < M && j < N && c < C) {
        int k = c + (j + i * N) * C;
        switch (op) {                                      /// no divergence
        case O_ADD: O[k] = A[k] + B[k]; break;
        case O_SUB: O[k] = A[k] - B[k]; break;
        case O_MUL: O[k] = A[k] * B[k]; break;             /// * convolution
        case O_DIV: O[k] = A[k] / B[k]; break;
        }
    }
}
__GPU__ Tensor&
Tensor::mm(
    Tensor &A, Tensor &B, Tensor &O, t4_mm_opt opt) {
    U16 M  = opt & MM_A_TXP ? A.W() : A.H();
    U16 Ka = opt & MM_A_TXP ? A.H() : A.W();
    U16 N  = opt & MM_B_TXP ? B.H() : B.W();
    U16 Kb = opt & MM_B_TXP ? B.W() : B.H();
    U16 C  = A.C();
    if (Ka != Kb || C != B.C()) {
        ERROR("Tensor#mm Ka(%d)!=Kb(%d) or C diff\n", Ka, Kb);
        return O;
    }
    WARN("Tensor#matmul C=%d, M=%d, N=%d, K=%d\n", C, M, N, Ka);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(N, M, C, blk));

    k_matmul<<<grd,blk>>>(A.data, B.data, O.data, M, N, Ka, opt);
    GPU_SYNC();
    
    return O;
}
/// tensor GEMM C' = alpha * A x B + beta * C
///
__GPU__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
    U16 M = A.H(), N = B.W(), Ka = A.W(), Kb = B.H(), C = A.C();
    if (Ka != Kb || C != B.C()) {
        ERROR("Tensor#gemm ka(%d)!=kb(%d) or C diff\n", Ka, Kb);
        return O;
    }
    WARN("GEMM C=%d, M=%d, N=%d, K=%d a=%f, b=%f\n", M, N, Ka, alpha, beta);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(N, M, C, blk));
    
    k_gemm<<<grd, blk>>>(A.data, B.data, O.data, M, N, Ka, alpha, beta);
    GPU_SYNC();
    
    return O;
}
///
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__GPU__ Tensor&
Tensor::matx(t4_ten_op op, Tensor &A, Tensor &B, Tensor &O) {
    U16 M = A.H(), N = A.W(), C = A.C();
    OPN("add", "sub", "mul", "div");
    WARN("Tensor::mat%s C=%d, M=%d, N=%d\n", opn[op], C, M, N);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(N, M, C, blk));
    
    k_mat_op<<<grd, blk>>>(op, A.data, B.data, O.data, M, N);
    GPU_SYNC();
    
    return O;
}
///
/// tensor-scalar addition O = A +- n element-wise (Hadamard)
///
__GPU__ Tensor&
Tensor::matx(t4_ten_op op, Tensor &A, DU v, Tensor &O) {
    U16 M = A.H(), N = A.W(), C = A.C();
    OPN("add", "sub", "mul", "div");
    WARN("Tensor::mat%s C=%d, M=%d, N=%d\n", opn[op], C, M, N);
    DU *d0 = O.data, *da = A.data;
    for (int k = 0; k < A.numel; k++) {
        switch (op) {
        case O_ADD: *d0++ = *da++ + v; break;
        case O_SUB: *d0++ = *da++ - v; break;
        case O_MUL: *d0++ = *da++ * v; break;
        case O_DIV: *d0++ = *da++ / v; break;
        }
    }
    return O;
}
__GPU__ Tensor&
Tensor::copy(Tensor &A, Tensor &O) {
    WARN("Tensor::copy numel=%d\n", A.numel);
    int n = (A.numel + T4_WARP_SQ - 1) / T4_WARP_SQ;
    
    k_copy<<<n, T4_WARP_SQ>>>(A.data, O.data, A.numel);
    GPU_SYNC();
    
    return O;
}
__GPU__ Tensor&
Tensor::transpose(Tensor &A, Tensor &T) {
    U16 M = A.H(), N = A.W(), C = A.C();
    WARN("Tensor::transpose A[%d,%d,%d]\n", M, N, C);
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(N, M, C, blk));
    
    k_transpose<<<grd, blk>>>(A.data, T.data, M, N);
    GPU_SYNC();
    
    return T;
}
///
/// matrix inversion (Gauss-Jordan with Pivot)
/// Note: Gauss-Jordan elimination is expensive O(N^3)
/// TODO: CDP
///
__GPU__ Tensor&
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
__GPU__ Tensor&
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
__GPU__ Tensor&
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
__GPU__ Tensor&
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
__GPU__ DU
Tensor::sum() {
    const int n = (numel + T4_WARP_SQ -1) / T4_WARP_SQ;
    DU *sum = new DU;
    *sum = DU0;

    k_sum<<<n, T4_WARP_SQ>>>(data, sum, numel);  /// * 8x straight loop
    GPU_SYNC();               /// * cooperative_groups.sync() does not work!
    
    DU v = *sum;
    delete sum;
    
    return SCALAR(v);
}
__GPU__ DU
Tensor::avg() {
    DU v = sum() / numel;
    return SCALAR(v);
}
__GPU__ DU
Tensor::std() {
    DU sum = DU0, avg = this->avg();
    DU *d  = data;
    for (int i=0; i < numel; i++, d++) {
        DU v = *d - avg;
        sum += v * v;
    }
    return numel ? SQRT(sum / numel) : DU0;
}
__GPU__ DU
Tensor::max() {
    DU v = data[0];
    for (int i=1; i < numel; i++) {              ///> TODO: CDP prefix sum
        v = MAX(data[i], v);
    }
    return SCALAR(v);
}
__GPU__ DU
Tensor::min() {
    DU v = data[0];
    for (int i=1; i < numel; i++) {              ///> TODO: CDP prefix sum
        v = MIN(data[i], v);
    }
    return SCALAR(v);
}
__GPU__ DU
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
__GPU__ DU
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
__GPU__ Tensor&
Tensor::triu() {
    U16 m  = H(), n = W();
    WARN("Tensor::upper[%d,%d]\n", m, n);

    for (U16 z = 1; z < m; z++) {
        for (U16 k = 0; k < z; k++) {
            data[k + z * n] = DU0;
        }
    }
    return *this;
}
///
/// matrix lower triangle with diag filled with 1
///
__GPU__ Tensor&
Tensor::tril() {
    U16 m = H(), n = W();
    WARN("Tensor::lower[%d,%d]\n", m, n);

    for (U16 z = 0; z < m; z++) {
        data[z + z * n] = DU1;
        for (U16 k = z + 1; k < n; k++) {
            data[k + z * n] = DU0;
        }
    }
    return *this;
}
///=======================================================================
/// Tensor life-cycle ops
///
__BOTH__ Tensor&
Tensor::reset(void *mptr, U32 sz, t4_obj tt, t4_layer fn) {
    WARN("Tensor::reset(%p, %d)\n", mptr, sz);
    init(sz, tt, 1);                                   /// T4Base attributes
    
    const U16 s[4] = { 1, 1, 1, 1 };
    const U16 t[4] = { (U16)sz, 1, 1, 1 };
    const DU  g[4] = { DU0, DU0, DU0, DU0 };
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
        WARN("Tensor::reshaped(%d,%d)\n", H(), W());
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
        WARN("Tensor::reshaped(%d,%d,%d,%d)\n", N(), H(), W(), C());
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
        WARN("Tensor::reshaped(%d,%d,%d,%d,%d)\n", c1, N(), H(), W(), C());
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
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(N, M, C(), blk));
    
    k_identity<<<grd, blk>>>(data, M, N, sizeof(DU));
    GPU_SYNC();
    
    return *this;
}

__BOTH__ Tensor&
Tensor::map(t4_ten_op op, DU v) {
    OPN("+", "-", "*", "/", "@", "x", "fill", "scale","pow", "abs", "exp", "log", "tanh", "relu", "sigmoid");
    WARN("Tensor#%s v=%f\n", opn[op], v);
    int n = (numel + T4_WARP_SQ - 1) / T4_WARP_SQ;
    
    k_ten_op<<<n, T4_WARP_SQ>>>(op, data, numel, v);
    GPU_SYNC();
    
    return *this;
}
