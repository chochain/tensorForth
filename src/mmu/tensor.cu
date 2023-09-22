/** -*- c++ -*-
 * @file
 * @brief Tensor class - ranked tensor impmementation i.e. vector, matrix, tensor, ...
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"

#if T4_ENABLE_OBJ

///=======================================================================
/// static methods
///
///> array sum
/// Note: tiled_partition<32> used
///
__KERN__ void k_sum(DU *I, DU *sum, int HW) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    DU vi = i < HW ? I[c + i * C + ns] : DU0;              ///< keep v for shuffle
    ///
    /// prefix sum (32-threaded tile)
    ///
    auto t = cg::tiled_partition<32>(cg::this_thread_block());
    auto shfl_sum = [](cg::thread_block_tile<32> t, DU v) {
        for (int k = 16; k > 0; k >>= 1) {
            v += t.shfl_down(v, k);
        }
        return v;
    };
    DU tt = shfl_sum(t, vi);
    ///
    /// sum up atomically (per channel for batchnorm)
    ///
    if (t.thread_rank() == 0) atomicAdd(&sum[c], tt);
}
///
///> variance
///
__KERN__ void
k_var(DU *I, DU *avg, DU *var, int HW) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    DU v0 = i < HW ? I[c + i * C + ns] - avg[c] : DU0;
    DU vi = v0 * v0;
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
    DU tt = shfl_sum(t, vi);
    ///
    /// sum up atomically (per channel, for batchnorm)
    ///
    if (t.thread_rank() == 0) atomicAdd_block(&var[c], tt);
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
/// Binary Cross-Entropy (clamps output to >= -100)
///
__KERN__ void
k_bce(DU *O, DU *T, int numel) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    if (i < numel) {
//        O[i] = ABS(T[i]) < DU_EPS ? LN(DU1 - O[i]) : LN(O[i]);
        O[i] = T[i] * LN(O[i]) + (DU1 - T[i]) * LN(DU1 - O[i]);
    }
}
///
/// tensor-scalar addition O = A op n element-wise (Hadamard)
///
__GPU__ Tensor&
Tensor::ten_op(math_op op, Tensor &A, DU v, Tensor &O) {
    U16 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    OPN("+", "-", "*", "/");
    WARN("Tensor::mat[%d,%d,%d,%d] %s %6.2f\n", N, H, W, C, opn[op], v);

    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((A.numel + blk.x - 1) / blk.x, 1, 1);
    
    k_ts_op<<<grd, blk>>>(op, A.data, v, O.data, A.numel);
    GPU_SYNC();
    
    return O;
}
///
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__GPU__ Tensor&
Tensor::ten_op(math_op op, Tensor &A, Tensor &B, Tensor &O) {
    U16 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    OPN(MATH_OP);
    WARN("Tensor::mat_%s[%d,%d,%d,%d]\n", opn[op], N, H, W, C);
    
    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((A.numel + blk.x - 1) / blk.x, 1, 1);
    
    k_tt_op<<<grd, blk>>>(op, A.data, B.data, O.data, A.numel);
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
        DU r0 = RCP(dd[z + z * n]);
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
    static DU sum; sum = DU0; ///< static shared memory

    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((numel + blk.x - 1)/blk.x, 1, 1);

    k_sum<<<grd, blk>>>(data, &sum, numel);  /// * 8x straight loop
    GPU_SYNC();               /// * cooperative_groups.sync() does not work!

    return SCALAR(sum);
}
__GPU__ DU
Tensor::avg() {
    DU v = sum() / numel;
    return SCALAR(v);
}
__GPU__ DU
Tensor::std() {
    static DU sum, avg;
    sum = DU0; avg = this->avg();

    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((numel + blk.x - 1)/blk.x, 1, 1);

    k_var<<<grd, blk>>>(data, &avg, &sum, numel);  /// * 8x straight loop
    GPU_SYNC();           /// * cooperative_groups.sync() does not work!
    
    DU v = numel ? SQRT(sum / numel) : DU0;
    
    return SCALAR(v);
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
__GPU__ DU
Tensor::loss(t4_loss op, Tensor &tgt) {
    /*
    auto check_bce = [this, &tgt]() {
        DU sum = DU0;
        for (int i=0; i<numel; i++) {
            DU t = tgt.data[i], y = this->data[i];
            sum += t * LN(y) + (DU1-t) * LN(DU1 - y);
        }
        return -sum;
    };
    */
    DU sum = DU0;                    ///> result loss value
    switch (op) {
    case LOSS_MSE:                   /// * mean squared error, input from linear
        *this -= tgt;
        sum = 0.5 * NORM(numel, data);
        break;
    case LOSS_BCE: {                 /// * binary cross_entropy, input from sigmoid
        dim3 blk(T4_WARP_SQ, 1, 1);
        dim3 grd((numel + blk.x - 1)/blk.x, 1, 1);
        k_bce<<<grd, blk>>>(data, tgt.data, numel);
        GPU_SYNC();
        sum = -this->sum();          /// * -(y * ln(out_i) + (1-y) * ln(1-out_i))
    } break;
    case LOSS_CE:                    /// * cross_entropy, input from softmax
        map(LN);                     /// * log(out_i)
        /* no break */
    case LOSS_NLL:                   /// * negative log likelihood, input from log-softmax
        *this *= tgt;                /// * out_i * tgt_i
        sum = -this->sum();          /// * negative sum
        break;
    default: ERROR("Model#loss op=%d not supported!\n", op);
    }
    sum /= numel;                    /// average per mini-batch sample
    
    return SCALAR(sum);              /// make sum a scalar value (not object)
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
    
    const U16    s[4] = { 1, 1, 1, 1 };
    const U16    h[4] = { (U16)sz, 1, 1, 1 };
    const Tensor *t[4]= { NULL, NULL, NULL, NULL };
    data    = (DU*)mptr;
    grad_fn = fn;
    memcpy(stride, s, sizeof(s));
    memcpy(shape,  h, sizeof(h));
    memcpy(grad,   t, sizeof(t));
    memcpy(mtum,   t, sizeof(t));
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
    const U16 s[4] = { 1, 1, 1, 1 }, t[4] = { h, w, 1, 1 };
    U32 sz = h * w;
    if (sz == numel) {
        rank = 2;
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
        rank = 4;
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
Tensor::map(math_op op, DU v) {
    OPN(MATH_OP);
    WARN("Tensor#%s v=%f\n", opn[op], v);
    int g = (numel + T4_WARP_SQ - 1) / T4_WARP_SQ;
    
    k_math<<<g, T4_WARP_SQ>>>(op, data, numel, v);
    GPU_SYNC();
    
    return *this;
}

__BOTH__ Tensor&
Tensor::normalize(DU avg, DU std) {
    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((numel + blk.x - 1) / blk.x, 1, 1);
    
    k_ts_op<<<grd, blk>>>(SUB, data, avg, data, numel);
    GPU_SYNC();
    k_ts_op<<<grd, blk>>>(DIV, data, std, data, numel);
    GPU_SYNC();

    return *this;
}
///=======================================================================
/// Tensor debugger
///
__BOTH__ void
Tensor::_dump(DU *v, int H, int W, int C) {
    const int hw = H * W, sq = (int)sqrt(hw);
    const int sh = (hw/sq) + ((hw - sq*sq) > 0 ? 1 : 0);
    const int h  = W > 1 ? H : (hw < 36 ? 1 : sh);
    const int w  = W > 1 ? W : (hw < 36 ? H : sq);
    
    DU *csum = new DU[C];
    for (int k = 0; k < C; k++) csum[k] = DU0;
    for (int i = 0; i < h; i++) {
        printf("\n");
        DU sum = DU0;
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < w; j++) {
                int n = j + i * w;
                if (n >= hw) { printf(" ...."); continue; }
                
                DU  r = v[k + n * C];
                printf("%5.2f", r);
                sum += r;
                csum[k] += r;
            }
            printf("|");
        }
        printf("Σ=%6.3f", sum);
    }
    if (h > 1) {
        printf("\nΣΣ=");
        for (int k = 0; k < C; k++) printf("%6.3f ", csum[k]);
    }
    delete csum;
}
///
///> _view - in ASCII art
///
__BOTH__ void
Tensor::_view(DU *v, int H, int W, int C, DU mean, DU scale) {
    auto map = [](DU v) {
        // static const char *lk = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";                             // 69 shades
        static const char *lk = " .:-=+*#%@X";      // 11 shades
        //return lk[v < 10.0f ? (v < DU0 ? 10 : (int)v) : 9];
        int i = static_cast<int>((v + 1.0) * 5.5);
        return lk[i < 0 ? 0 : (i > 10 ? 10 : i)];
    };
    const int hw = H * W, sr = static_cast<int>(sqrtf(hw));
    const int sh = (hw/sr) + ((hw - sr*sr) > 0 ? 1 : 0);
    const int w  = W > 1 ? W : (hw < 36 ? H : sr);
    const int h  = W > 1 ? H : (hw < 36 ? 1 : sh);

    DU *csum = new DU[C];
    for (int k = 0; k < C; k++) csum[k] = DU0;
    for (int i = 0; i < h; i++) {
        printf("\n");
        for (int k = 0; k < C; k++) {
            for (int j = 0; j < w; j++) {
                int n = j + i * w;
                if (n >= hw) { printf("  "); continue; }
                
                DU r0 = v[k + (j>0 ? n - 1 : n) * C];
                DU r1 = v[k + n * C];
                DU x0 = (r0 - mean) * scale;
                DU x1 = (((r0 + r1) * 0.5) - mean) * scale;

                printf("%c%c", map(x0), map(x1));  // double width
                csum[k] += r1;
            }
            printf("|");
        }
    }
    if (h > 1) {
        printf("\nΣΣ=");
        for (int k = 0; k < C; k++) printf("%6.3f ", csum[k]);
    }
    printf("\n");
    
    delete csum;
}

__GPU__ void
Tensor::show(bool dump) {
    const U16 N  = this->N(), H = this->H(), W = this->W(), C = this->C();
    const int hw = H * W;

    DU mean  = avg();
    DU scale = 0.5 / std();            // P=95%
    for (int n = 0; n < N; n++) {
        DU *d = slice(n);
        if (dump || hw < 100) {
            printf("\nn=%d", n);
            _dump(d, H, W, C);
        }
        if (hw > 36) _view(d, H, W, C, mean, scale);
    }
    printf("\n");
}

#endif // T4_ENABLE_OBJ
