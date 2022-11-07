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
        case O_ADD:   t[k] += v;                        break;
        case O_SUB:   t[k] -= v;                        break;
        case O_MUL:   t[k] *= v;                        break;
        case O_DIV:   t[k] /= v;                        break;
        case O_FILL:  t[k] = v;                         break;
        case O_SCALE: t[k] *= v;                        break;
        case O_POW:   t[k] = POW(t[k], v);              break;
        case O_ABS:   t[k] = ABS(t[k]);                 break;
        case O_EXP:   t[k] = EXP(t[k]);                 break;
        case O_LOG:   t[k] = LOG(t[k]);                 break;
        case O_TANH:  t[k] = TANH(t[k]);                break;
        case O_RELU:  t[k] = t[k] > DU0 ? t[k] : DU0;   break;
        case O_SIGM:  t[k] = DU1 / (DU1 + EXP(-t[k]));  break;
        default: ERROR("k_ten_op %d not supported\n", op);
        }
    }
}
///
/// array sum
/// Note: tiled_partition<32> used
///
__KERN__ void
k_sum(DU *A, DU *sum, int sz) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    DU vj = j < sz ? A[j] : DU0;
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
    if (t.thread_rank() == 0) atomicAdd_block(sum, tt);
}
__KERN__ void
k_matmul(
    DU *A, DU *B, DU *O,   /* O[NxHxWxC] = A[NxHxKxC] @ B[NxKxWxC] */
    int H, int W, int K,
    t4_mm_opt opt)
{
    const int i  = threadIdx.y + blockIdx.y * blockDim.y;  ///< H
    const int j  = threadIdx.x + blockIdx.x * blockDim.x;  ///< W
    const int c  = blockIdx.z,  C = gridDim.z;             ///< C
    const int z0 = c + (j + i * W) * C;                    ///< output matrix index
    
    if (i < H && j < W && c < C) {                         /// * TODO: tiled
        DU  *ax, *bx;
        int ai, bi;
        if (opt & MM_A_TXP) {                              /// * transpose A
            ax = &A[c + i * C];     ai = H * C;
            bx = &B[c + j * C];     bi = W * C;
        }
        else if (opt & MM_B_TXP) {                         /// * transpose B
            ax = &A[c + i * K * C]; ai = C;
            bx = &B[c + j * K * C]; bi = C;
        }
        else {                                             /// * no tranposition
            ax = &A[c + i * K * C]; ai = C;
            bx = &B[c + j * C];     bi = W * C;
        }
        DU2 acc = DU0;                                     /// * TODO: suffle sum
//      acc += ax[k * C] * bx[k * N * C];                  /// * 8.1 ms 1Kx1K
        for (int k = 0; k < K; k++, ax += ai, bx += bi) {
            acc += (*ax) * (*bx);                          /// * 6.2 ms 1Kx1K
        }
        if (opt & MM_INC) O[z0] += acc;                    /// * increment O
        else              O[z0] =  acc;                    /// * overwrite O
    }
}
///
/// GEMM kernel (used CUDA dynamic parallelism)
///     O = alpha * A x B + beta * O
///     where A = HxKxC, B = KxWxC, O = HxWxC
///
__KERN__ void
k_gemm(
    DU *A, DU *B, DU *O,  /* O[HxWxC] = a * A[HxKxC] @ B[KxWxC] + b * O[HxWxC] */
    int H, int W, int K,
    DU alpha, DU beta)
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;   ///< H
    const int j = threadIdx.x + blockIdx.x * blockDim.x;   ///< W
    const int c = blockIdx.z, C = gridDim.z;               ///< channel deep
    const int WC= W * C;
    const int z0= c + (j + i * W) * C;                     ///< output index

    if (i < H && j < W && c < C) {                         /// * TODO: tiled
        DU *ax = &A[c + i * K * C];
        DU *bx = &B[c + j * C];
        DU2 acc = DU0;                                     /// * TODO: suffle sum
        for (int k = 0; k < K; k++, ax += C, bx += WC) {
            acc += (*ax) * (*bx);
        }
        O[z0] = alpha * acc + beta * O[z0];                /// * scaling
    }
}
///
/// matrix-matrix element-wise ops
///
__KERN__ void k_tt_op(
    t4_ten_op op,
    DU *A, DU *B, DU *O,
    int HW)
{
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index

    if (i < HW && c < C) {
        switch (op) {                                     /// no divergence
        case O_ADD: O[k] = A[k] + B[k]; break;
        case O_SUB: O[k] = A[k] - B[k]; break;
        case O_MUL: O[k] = A[k] * B[k]; break;            /// * convolution
        case O_DIV: O[k] = A[k] / B[k]; break;
        }
    }
}
__KERN__ void k_ts_op(
    t4_ten_op op,
    DU *A, DU v, DU *O,
    int HW)
{
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index

    if (i < HW && c < C) {
        switch (op) {                                      /// no divergence
        case O_ADD: O[k] = A[k] + v; break;
        case O_SUB: O[k] = A[k] - v; break;
        case O_MUL: O[k] = A[k] * v; break;                /// * convolution
        case O_DIV: O[k] = A[k] / v; break;
        }
    }
}
///
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__GPU__ Tensor&
Tensor::ten_op(t4_ten_op op, Tensor &A, Tensor &B, Tensor &O) {
    U16 N = A.N(), H = A.H(), W = A.W(), C = A.C(), HW = H * W;
    
    OPN("add", "sub", "mul", "div");
    WARN("Tensor::mat%s[%d,%d,%d,%d]\n", opn[op], N, H, W, C);
    
    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((HW + blk.x - 1) / blk.x, C, N);
    
    k_tt_op<<<grd, blk>>>(op, A.data, B.data, O.data, HW);
    GPU_SYNC();
    
    return O;
}
///
/// tensor-scalar addition O = A +- n element-wise (Hadamard)
///
__GPU__ Tensor&
Tensor::ten_op(t4_ten_op op, Tensor &A, DU v, Tensor &O) {
    U16 N = A.N(), H = A.H(), W = A.W(), C = A.C(), HW = H * W;

    OPN("add", "sub", "mul", "div");
    WARN("Tensor::mat[%d,%d,%d,%d] %s %6.2f\n", N, H, W, C, opn[op], v);

    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((HW + blk.x - 1) / blk.x, C, N);
    
    k_ts_op<<<grd, blk>>>(op, A.data, v, O.data, HW);
    GPU_SYNC();
    
    return O;
}
__GPU__ Tensor&
Tensor::mm(
    Tensor &A, Tensor &B, Tensor &O, t4_mm_opt opt) {
    U16 H  = opt & MM_A_TXP ? A.W() : A.H();
    U16 Ka = opt & MM_A_TXP ? A.H() : A.W();
    U16 W  = opt & MM_B_TXP ? B.H() : B.W();
    U16 Kb = opt & MM_B_TXP ? B.W() : B.H();
    U16 N  = B.N(), C = B.C();                     /// B, O common dimensions
    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("Tensor#mm Ka(%d)!=Kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    WARN("Tensor#matmul N=%d, C=%d, H=%d, W=%d, K=%d\n", N, C, H, W, Ka);
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(NGRID(W, H, C, blk));

    for (int n = 0; n < N; n++) {
        DU *da = A.data, *db = B.slice(n), *dx = O.slice(n);
        k_matmul<<<grd,blk>>>(da, db, dx, H, W, Ka, opt);
    }
    GPU_SYNC();
    
    return O;
}
///
/// tensor GEMM C' = alpha * A x B + beta * C
///
__GPU__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
    U16 H = A.H(), W = B.W(), Ka = A.W(), Kb = B.H();
    U16 N = B.N(), C = B.C();
    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("Tensor#gemm ka(%d)!=kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    WARN("GEMM N=%d, C=%d, H=%d, W=%d, K=%d a=%f, b=%f\n", N, C, H, W, Ka, alpha, beta);
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(NGRID(W, H, C, blk));

    for (int n = 0; n < N; n++) {
        DU *da = A.data, *db = B.slice(n), *dx = O.slice(n);
        k_gemm<<<grd, blk>>>(da, db, dx, H, W, Ka, alpha, beta);
    }
    GPU_SYNC();
    
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
    U16 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    WARN("Tensor::transpose A[%d,%d,%d,%d]\n", N, H, W, C);
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(NGRID(W, H, C, blk));

    for (int n = 0; n < N; n++) {
        DU *da = A.slice(n), *dt = T.slice(n);
        k_transpose<<<grd, blk>>>(da, dt, H, W);
    }
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
    sum = numel ? SQRT(sum / numel) : DU0;
    return SCALAR(sum);
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
    return SCALAR(acc);
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

    return SCALAR(v);
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
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(NGRID(W(), H(), C(), blk));

    for (int n = 0; n < N(); n++) {
        k_identity<<<grd, blk>>>(slice(n), H(), W(), sizeof(DU));
    }
    GPU_SYNC();
    
    return *this;
}

__BOTH__ Tensor&
Tensor::map(t4_ten_op op, DU v) {
    OPN("+", "-", "*", "/", "@", "solv", "fill", "scale","pow", "abs", "exp", "log", "tanh", "relu", "sigmoid");
    WARN("Tensor#%s v=%f\n", opn[op], v);
    int n = (numel + T4_WARP_SQ - 1) / T4_WARP_SQ;
    
    k_ten_op<<<n, T4_WARP_SQ>>>(op, data, numel, v);
    GPU_SYNC();
    
    return *this;
}
