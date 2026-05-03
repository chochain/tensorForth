/** -*- c++ -*-
 * @file
 * @brief Tensor class - ranked tensor impmementation i.e. vector, matrix, tensor, ...
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <float.h>      // FLT_MAX
#include "tensor.h"

namespace t4::mu {

#if T4_DO_OBJ
///
/// tensor-scalar addition O = A op n element-wise (Hadamard)
///
__HOST__ Tensor&
Tensor::ten_op(math_op op, Tensor &A, DU v, Tensor &O) {
    U32 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    _OP(MATH_OP);
    MM_DB("  tensor#ten_op O[%d,%d,%d,%d] = A %s %6.2f\n", N, H, W, C, _op[op], v);
    FORK(k_ts_op, A.numel, op, A.data, v, O.data);
    return O;
}
///
/// tensor-tensor element-wise C = A op B where op=ADD|SUB|MUL|DIV (Hadamard)
///
__HOST__ Tensor&
Tensor::ten_op(math_op op, Tensor &A, Tensor &B, Tensor &O) {
    U32 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    _OP(MATH_OP);
    MM_DB("  tensor#ten_op O[%d,%d,%d,%d] = A %s B\n", N, H, W, C, _op[op]);
    FORK(k_tt_op, A.numel, op, A.data, B.data, O.data);
    return O;
}
// ---------------------------------------------------------------------------
// dot — host wrapper, batched over N samples and C channels
//
//   A, B : rank-2 tensors  [H=1, W=K, C, N]
//   O    : rank-1 tensor   [H=1, W=1, C, N]  (one scalar per channel per sample)
// ---------------------------------------------------------------------------
__HOST__ Tensor&
Tensor::dot(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
    const U32 K = A.W(), C = A.C(), N = A.N();   ///< vector length, channels, batch_size

    MM_DB("  tensor#dot O[%d,%d,%d,%d] = %g A @ B + %g O\n", N, 1, K, C, alpha, beta);
    for (U32 n = 0; n < N; n++) {
        DU *da = A.slice(n), *db = B.slice(n), *dx = O.slice(n);
        FORK1(k_dot, C, 1, da, db, dx, alpha, beta, K, C);
    }
    return O;
}
__HOST__ Tensor&
Tensor::mm(
    Tensor &A, Tensor &B, Tensor &O, bool inc, bool tA, bool tB) {
    return gemm3(A, B, O, DU1, inc ? DU1 : DU0, tA, tB);
}

__HOST__ Tensor&
Tensor::linear(
    Tensor &A, Tensor &B, Tensor &O, int H, int W, int K, DU alpha, DU beta, bool tA, bool tB) {
    MM_DB("  tensor#linear a=%g, b=%g => O[%d,%d] = A%s[%d,%d] @ B%s[%d,%d]\n",
          alpha, beta, H, W, tA ? "^T" : "", H, K, tB ? "^T" : "", K, W);

    FORK3T(k_gemm_tile_claude, H, W, 1, A.data, B.data, O.data, alpha, beta, tA, tB, K);
    return O;
}
///
/// tensor GEMM C' = alpha * A x B + beta * C
/// @note - benchmark alpha * [1K,1K]*[1K,1K] + beta [1K,1K]
///   0: x86_gemm              - 2227.2 ms
///   1: k_gemm                -   30.7 ms
///   2: k_gemm_claude         -   28.1 ms
///   3: k_gemm_tile_claude    -   6.4~3.2 ms
///   4: k_gemm_tile_claude_x2 -    3.2 ms
///
__HOST__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta, bool tA, bool tB) {
    U32 H = A.H(), W = B.W(), Ka = A.W(), Kb = B.H();
    U32 N = B.N(), C = B.C();
    
    const int BLOCK = 32;
    for (int i = 0; i < H*W; ++i) O[i] *= beta;     /// * apply beta, 0.0f zero out C

    for (int kk = 0; kk < Ka; kk += BLOCK) {        /// * accumulate alpha * (A * B)
        for (int mm = 0; mm < H; mm += BLOCK) {
            for (int nn = 0; nn < W; nn += BLOCK) {
                
                for (int k = kk; k < MIN(kk + BLOCK, Ka); ++k) {
                    for (int i = mm; i < MIN(mm + BLOCK, H); ++i) {
                        /// pre-multiply alpha with the A element to save operations
                        float av = alpha * A[i * Ka + k]; 
                        for (int j = nn; j < MIN(nn + BLOCK, W); ++j) {
                            O[i * W + j] += av * B[k * W + j];
                        }
                    }
                }
                
            }
        }
    }
    return O;
}

__HOST__ Tensor&
Tensor::gemm1(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta, bool tA, bool tB) {
    U32 H = A.H(), W = B.W(), Ka = A.W(), Kb = B.H();
    U32 N = B.N(), C = B.C();
    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("  tensor#gemm1 ka(%d)!=kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    MM_DB("  tensor#gemm1 K=%d, a=%g, b=%g => NHWC=[%d,%d,%d,%d]\n",
          Ka, alpha, beta, N, H, W, C);

    for (U32 n = 0; n < N; n++) {
        DU *da = A.slice(n), *db = B.slice(n), *dx = O.slice(n);
        FORK3(k_gemm, H, W, C, da, db, dx, alpha, beta, tA, tB, Ka);
    }
    return O;
}
__HOST__ Tensor&
Tensor::gemm2(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta, bool tA, bool tB) {
    U32 H = A.H(), W = B.W(), Ka = A.W(), Kb = B.H();
    U32 N = B.N(), C = B.C();
    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("  tensor#gemm2 ka(%d)!=kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    MM_DB("  tensor#gemm2 K=%d, a=%g, b=%g => NHWC=[%d,%d,%d,%d]\n",
          Ka, alpha, beta, N, H, W, C);

    for (U32 n = 0; n < N; n++) {
        DU *da = A.slice(n), *db = B.slice(n), *dx = O.slice(n);
        FORK3(k_gemm_claude, H, W, C, da, db, dx, alpha, beta, tA, tB, Ka);
    }
    return O;
}
__HOST__ Tensor&
Tensor::gemm3(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta, bool tA, bool tB) {
    U32 H  = tA ? A.W() : A.H(), W  = tB ? B.H() : B.W();
    U32 Ka = tA ? A.H() : A.W(), Kb = tB ? B.W() : B.H();
    U32 N  = B.N(), C = B.C();

    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("  tensor#gemm3 ka(%d)!=kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    MM_DB("  tensor#gemm3 K=%d, a=%g, b=%g => NHWC=[%d,%d,%d,%d]\n",
          Ka, alpha, beta, N, H, W, C);

    for (U32 n = 0; n < N; n++) {
        DU *da = A.slice(n), *db = B.slice(n), *dx = O.slice(n);
        FORK3T(k_gemm_tile_claude, H, W, C, da, db, dx, alpha, beta, tA, tB, Ka);
    }
    return O;
}
__HOST__ Tensor&
Tensor::gemm4(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta, bool tA, bool tB) {
    U32 H  = tA ? A.W() : A.H(), W  = tB ? B.H() : B.W();
    U32 Ka = tA ? A.H() : A.W(), Kb = tB ? B.W() : B.H();
    //       0  ? E0    : 1           1  ? E0    : E1
    U32 N  = B.N(), C = B.C();

    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("  tensor#gemm4 ka(%d)!=kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    MM_DB("  tensor#gemm4 K=%d, a=%g, b=%g => NHWC=[%d,%d,%d,%d]\n",
          Ka, alpha, beta, N, H, W, C);

    for (U32 n = 0; n < N; n++) {
        DU *da = A.slice(n), *db = B.slice(n), *dx = O.slice(n);
        FORK3T(k_gemm_tile_claude_x2, H, W, C, da, db, dx, alpha, beta, tA, tB, Ka);
    }
    return O;
}

__HOST__ Tensor&
Tensor::copy(Tensor &A, Tensor &O) {
    MM_DB("  tensor#copy %p to %p numel=%ld\n", A.data, O.data, A.numel);
    FORK(k_copy, A.numel, A.data, O.data);
    return O;
}
__HOST__ Tensor&
Tensor::transpose(Tensor &A, Tensor &T) {
    U32 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    MM_DB("  tensor#transpose A[%d,%d,%d,%d]\n", N, H, W, C);
    
    for (U32 n = 0; n < N; n++) {
        DU *da = A.slice(n), *dt = T.slice(n);
        FORK3(k_transpose, H, W, C, da, dt);
    }
    return T;
}
__HOST__ Tensor&
Tensor::batchsum(Tensor &A, Tensor &O) {
    U32 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    MM_DB("  tensor#batchsum A[%d,%d,%d,%d] => O[%d, %d]\n", N, H, W, C, N, C);
    O.zeros();
    FORK4(k_batchsum, A.data, O.data, (U64)H*W);
    return O;
}
__HOST__ Tensor&
Tensor::batchvar(Tensor &A, Tensor &G, Tensor &O) {
    U32 N = A.N(), H = A.H(), W = A.W(), C = A.C();
    U64 NHW = (U64)N*H*W;
    MM_DB("  tensor#batchvar A[%d,%d,%d,%d] => O[%d,%d]\n", N, H, W, C, N, C);
    batchsum(A, G);
    G *= DU1 / NHW;
    O.zeros();
    FORK4(k_batchnvar, A.data, G.data, O.data, (U64)H*W);

    for (int i=0; i< O.numel; i++) {
        O.data[i] = SQRT(O.data[i] / NHW);
    }
    return O;
}

///=======================================================================
/// tensor arithmetics
///
__HOST__ DU
Tensor::sum() {
    DU v = DU0;
    if (numel < T4_DIM_SZ) {                           /// * cheaper for small loop
        for (int i = 0; i < numel; i++) v += data[i];
    }
    else {
        H2D(_tmp, &v, sizeof(DU));                     /// * pre-zero _tmp
        FORK(k_sum, numel, data, _tmp);                /// * data[numel] for temp storage
        D2H(&v, _tmp, sizeof(DU));                     /// * copy to host (D2H, page fault)
    }
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::avg() {
    DU v = sum() / numel;
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::std() {
    DU v = DU0, mx = avg();
    H2D(_tmp, &v, sizeof(DU));                         /// * pre-zero temp
    FORK(k_nvar, numel, data, mx, _tmp);               /// * 8x straight loop
    D2H(&v, _tmp, sizeof(DU));                         /// * copy to host (D2H, page fault)
    v = numel ? SQRT(v) / numel : DU0;             
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::norm() {                                       ///< return Euclidean Norm
    DU v = DU0;
    H2D(_tmp, &v, sizeof(DU));                         /// * pre-zero _tmp
    FORK(k_nvar, numel, data, DU0, _tmp);              /// * 8x straight loop
    D2H(&v, _tmp, sizeof(DU));                         /// * copy to host (D2H, page fault)
    v = SQRT(*_tmp);                                   
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::max() {
    const DU x = -FLT_MAX;
    H2D(_tmp, &x, sizeof(DU));                         /// * pre-set min, copy host to device
    FORK(k_max, numel, data, _tmp, true);              /// * find max
    DU v;
    D2H(&v, _tmp, sizeof(DU));                         /// * copy back to host
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::min() {
    const DU x = FLT_MAX;
    H2D(_tmp, &x, sizeof(DU));                         /// * pre-set max, copy host to device
    FORK(k_max, numel, data, _tmp, false);             /// * find min
    DU v;
    D2H(&v, _tmp, sizeof(DU));                         /// * copy back to host
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::dot(Tensor &B) {
    if (rank == 1 && B.rank == 1 && numel == B.numel) {
        FORK1(k_dot, 1, 1, data, B.data, _tmp, DU1, DU0, numel, 1);
    }
    else ERROR("A.dot(B) dim? %ld != %ld)\n", numel, B.numel);
    DU v;
    D2H(&v, _tmp, sizeof(DU));                        ///< copy back to host
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::loss(t4_loss op, Tensor &tgt) {
/*    
    auto check_bce = [this, &tgt]() {
        DU sum = DU0;
        for (int i=0; i<numel; i++) {
            DU t = tgt.data[i], y = this->data[i];
            sum += t * LN(y + DU_EPS) + (DU1-t) * LN(DU1 - y + DU_EPS);
        }
        return -sum;
    };
*/
    DU z = DU0;                      ///> result loss value
    switch (op) {
    case LOSS_MSE:                   /// * mean squared error, input from linear
        *this -= tgt;                /// * (output - predict)
        *this *= *this;              /// * (output - predict)^2
        z = sum();
        break;
    case LOSS_BCE: {                 /// * binary cross_entropy, input from sigmoid
        FORK(k_bce, numel, tgt.data, data);
        z = -sum();                  /// * -(y * ln(out_i) + (1-y) * ln(1-out_i))
    } break;
    case LOSS_CE:                    /// * cross_entropy, input from softmax
        map(LN);                     /// * log(out_i)
        /* no break */
    case LOSS_NLL:                   /// * negative log likelihood, input from log-softmax
        *this *= tgt;                /// * out_i * tgt_i
        z = -sum();                  /// * sum for mini-batch samples
        break;
    default: ERROR("Model#loss op=%d not supported!\n", op);
    }
    z /= N();                        /// * mini-batch average
    
    SCALAR(z); return z;             /// make sum a scalar value (not object)
}
__HOST__ U32
Tensor::has_nan() {
    int cnt = 0;
    H2D(_tmp, &cnt, sizeof(U32));
    FORK(k_nan_inf, numel, data, (int*)_tmp);
    D2H(&cnt, (int*)_tmp, sizeof(int));
    return cnt;
}

///=======================================================================
/// linear algebra methods
///=======================================================================
///
/// inverse — Gauss-Jordan matrix inversion
///
/// @param A  square input matrix  [K x K], column-major (k + z * K)
/// @param I  identity matrix in,  inverted matrix out  [K x K]
///
__HOST__ Tensor&
Tensor::inverse(Tensor &A, Tensor &I) {          /// * A=>I, I=>A^-1
    if (!A.is_square("A") || !I.is_square("I")) return A;
    
    const int K = A.W();
    INFO("  tensor#inverse [%d,%d]\n", K, K);
    
    int *d_pivot = (int*)A._tmp;                 /// * pivot result
    DU  *da = A.data, *di = I.data;
    
    for (int z = 0; z < K; z++) {
        int u = 0;
        FORK2(k_find_pivot, K, da, d_pivot, z);  /// * find pivot row
        D2H(&u, d_pivot, sizeof(int));
        if (u < 0) {
            ERROR("  tensor#inverse: singular matrix at column %d\n", z);
            return A;
        }
        if (u != z) {
            FORK(k_swap_rows, K, da, di, u, z);  /// * swap rows z ↔ u 
        }
        FORK(k_diag, K, da, di, z);              /// * normalise pivot row
        FORK(k_elim, K, da, di, z);              /// * eliminate column z from all other rows
    }
    return I;                                    /// * A becomes identity matrix
}

__HOST__ Tensor&
Tensor::plu(Tensor &A, Tensor &I, int *d_piv) {  ///< update A -> PLU (in-place)
    if (!A.is_square("A")) return A;
    
    const int K = A.W();
    int *d_pivot = (int*)A._tmp;                 ///< scalar: current pivot row
    DU  *da = A.data, *di = I.data;
    // -------------------------------------------------------------------------
    // Stage 1 — k_getrf: factorise A → P·L·U  (in-place, n sync points)
    // -------------------------------------------------------------------------
    for (int z = 0; z < K; z++) {
        int u = -1;
        FORK2(k_find_pivot, K, da, d_pivot, z);
        d_piv[z] = *d_pivot;                     ///< D2D, for Stage 2 (lu_inverse)
        D2H(&u, d_pivot, sizeof(int));           /// * capture on host
        
        if (u < 0) {
            ERROR("  tensor#plu: singular at column %d\n", z);
            return A;
        }

        if (u != z) FORK(k_swap_rows, K, da, nullptr, u, z);
        FORK(k_lu_col, K, da, z);
    }
    if (A!=I) FORK(k_pivot, K, da, d_piv, di);
    
    return I;                                    /// * A = L\U, I =>P, d_piv (permutation table)
}

__HOST__ Tensor&
Tensor::lu_inverse(Tensor &A, Tensor &I, int *d_piv) {
    if (!A.is_square("A") || !I.is_square("I")) return I;

    int K = A.W();
    INFO("  tensor#lu_inverse [%d,%d]\n", K, K);
    
    plu(A, I, d_piv);                            /// * A -> L\U, I => P, d_piv (in-place)

    // -------------------------------------------------------------------------
    // Stage 2 — k_getri: solve A·X = I  (only 2 kernel launches, fully parallel)
    // -------------------------------------------------------------------------
    DU  *da = A.data, *di = I.data;
    FORK(k_fsub, K, da, di);                     /// * L·Y = P·I
    FORK(k_bsub, K, da, di);                     /// * U·X = Y
    
    return I;
}

__HOST__ Tensor&
Tensor::lu(Tensor &LU, bool get_u) {
    if (!LU.is_square("LU")) return LU;

    const int K = LU.H();
    MM_DB("  tensor#%s [%d,%d]\n", get_u ? "upper" : "lower", K, K);

    FORK3(k_lu, K, K, 1, LU.data, get_u);         /// * turn LU to either L or U

    return LU;
}

__HOST__ DU
Tensor::det() {
    const U32 K = H();

    int piv[K], *d_piv;                           ///< permutation flags
    MM_ALLOC(&d_piv, sizeof(int) * K);            ///< d_piv[K], d_pivot[1], d_sign[1]
    
    plu(*this, *this, d_piv);                     ///< A → P·L·U in-place (*this, *this => no I update)
    D2H(piv, d_piv, sizeof(int) * K);             /// * move to host

    int cnt = 0;                                  /// count row swaps for det(P) sign
    for (int i = 0; i < K; i++)
        if (piv[i] != i) cnt++;                   ///< each entry = one actual swap
    const int sign = (cnt % 2 == 0) ? 1 : -1;
    
    DU det;                                       /// product of U diagonal (in log space for stability)
    FORK2(k_logdet, K, data, _tmp, d_piv);        /// * calculate log(determinant)
    D2H(&det, _tmp, sizeof(DU));                  /// * capture d_det 
    D2H(&cnt, d_piv, sizeof(int));                /// * capture d_sign
    
    MM_FREE(d_piv);

    det = EXP(det) * sign * cnt;                  /// * calculate determinant
    SCALAR(det); return det;
}
///=======================================================================
/// Tensor life-cycle ops
///
__HOST__ Tensor&
Tensor::reset(void *mem, U64 sz, t4_obj tt, t4_layer fn) {
    MM_DB("  tensor#reset(%p,%ld)\n", mem, sz);
    init(sz, tt, 1);                                  /// T4Base attributes

    const U64 GB   = 1L << 30;
    const U16 s[4] = { 1, 1, 1, 1 };
    const U32 h[4] = {
        (U32)(sz > GB ? (sz>>30) : sz),
        (U32)(sz > GB ? GB : 1L),
        1, 1
    };
    const Tensor *t[5]= { NULL, NULL, NULL, NULL, NULL };
    data    = (DU*)mem;
    grad_fn = fn;
    memcpy(stride, s, sizeof(s));
    memcpy(shape,  h, sizeof(h));
    memcpy(grad,   t, sizeof(t));
    memcpy(mtum,   t, sizeof(t));
    _tmp = &data[numel];                             /// * point tmp stroage to data[numel]
    
    return *this;
}

__HOST__ Tensor&
Tensor::reshape(U64 sz) {
    if (sz == numel) {
        reset(data, numel, (t4_obj)ttype, grad_fn);   /// preserve ttype and fn
        MM_DB("  tensor#reshaped(%ld)\n", numel);
    }
    else {
        ERROR("  tensor#reshape sz != numel (%ld != %ld)\n", sz, numel);
    }
    return *this;
}

__HOST__ Tensor&
Tensor::reshape(U32 h, U32 w) {
    const U16 s[4] = { 1, 1, 1, 1 };
    const U32 t[4] = { h, w, 1, 1 };
    U64 sz = (U64)h * w;
    if (sz == numel) {
        rank = 2;
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        MM_DB("  tensor#reshaped(%d,%d)\n", H(), W());
    }
    else {
        ERROR("  tensor#reshape sz != numel (%ld != %ld)\n", sz, numel);
    }
    return *this;
}

__HOST__ Tensor&
Tensor::reshape(U32 n, U32 h, U32 w, U32 c) {
    const U16 s[4] = { 1, 1, 1, 1 };
    const U32 t[4] = { h, w, c, n };
    U64 sz = (U64)n * h * w * c;
    if (sz == numel) {
        rank = 4;
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        MM_DB("  tensor#reshaped(%d,%d,%d,%d)\n", N(), H(), W(), C());
    }
    else {
        ERROR("  tensor#reshape sz != numel (%ld != %ld)\n", sz, numel);
    }
    return *this;
}
__HOST__ Tensor&
Tensor::reshape(U32 c1, U32 n, U32 h, U32 w, U32 c) {
    const U16 s[4] = { 1, 1, 1, 1 };
    const U32 t[4] = { h, w, c, n };
    U64 sz = (U64)c1 * n * h * w * c;
    if (sz == numel) {
        rank  = 5;
        iparm = c1;        /// use iparm field, so we don't need s[5]
        memcpy(stride, s, sizeof(s));
        memcpy(shape,  t, sizeof(t));
        MM_DB("  tensor#reshaped(%d,%d,%d,%d,%d)\n", c1, N(), H(), W(), C());
    }
    else {
        ERROR("  tensor#reshape sz != numel (%ld != %ld)\n", sz, numel);
    }
    return *this;
}

__HOST__ Tensor&
Tensor::identity() {
    const U32 W = this->W(), H = this->H(), C = this->C();
    for (U32 n = 0; n < N(); n++) {
        FORK3(k_identity, H, W, C, slice(n));
    }
    return *this;
}

__HOST__ Tensor&
Tensor::zeros() {
    cudaMemset(data, 0, sizeof(DU) * numel);
    GPU_CHK();
    return *this;
}

__HOST__ Tensor&
Tensor::map(math_op op, DU v) {
    _OP(MATH_OP);
    MM_DB("  tensor#%s v=%g\n", _op[op], v);
    FORK(k_math, numel, op, data, v);
    return *this;
}

__HOST__ Tensor&
Tensor::normalize(DU avg, DU std) {
    FORK(k_ts_op, numel, SUB, data, avg, data);
    FORK(k_ts_op, numel, DIV, data, std, data);
    return *this;
}
///=======================================================================
/// Tensor debugger
///
__HOST__ void
Tensor::_dump(DU *v, U32 H, U32 W, U32 C) {
    const DU  hw = I2D(H) * W, sr = sqrtf(hw);
    const U32 sh = UINT(hw / sr) + ((hw - sr*sr) > DU0 ? 1 : 0);
    const U32 h  = W > 1 ? H : (hw < 36.0 ? 1 : sh);
    const U32 w  = W > 1 ? W : (hw < 36.0 ? H : UINT(sr));
    
    DU *csum = new DU[C];
    for (U32 k = 0; k < C; k++) csum[k] = DU0;
    for (U32 i = 0; i < h; i++) {
        INFO("\n");
        DU sum = DU0;
        for (U32 k = 0; k < C; k++) {
            for (U32 j = 0; j < w; j++) {
                U64 n = j + i * w;
                if (n >= hw) { INFO(" ...."); continue; }
                
                DU  r = v[k + n * C];
                INFO("%5.2f", r);
                sum += r;
                csum[k] += r;
            }
            INFO("|");
        }
        INFO("Σ=%6.3f", sum);
    }
    if (h > 1) {
        INFO("\nΣΣ=");
        for (U32 k = 0; k < C; k++) INFO("%6.3f ", csum[k]);
    }
    delete csum;
}
///
///> _view - in ASCII art
///
__HOST__ void
Tensor::_view(DU *v, U32 H, U32 W, U32 C, DU mean, DU scale) {
    auto map = [](DU v) {
        // static const char *lk = " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";          /// 91 shades
        // static const char *lk = " `.-:^=;i>+!*zsv7C3tno5xakhdOUAXR#$0MW%Q"; /// 40 shades
        static const char *lk = " `.-:;!+*ixekO#@";     /// 16 shades
        static const int   sz = 16;
        int i  = static_cast<int>((v + 1.0) * sz/2);
        return lk[i < 0 ? 0 : (i < sz ? i : sz-1)];
    };
    const U64 hw = H * W, sr = static_cast<U64>(sqrtf(hw));
    const U32 sh = (hw / sr) + ((hw - sr*sr) > 0L ? 1 : 0);
    const U32 w  = W > 1 ? W : (hw < 36L ? H : sr);
    const U32 h  = W > 1 ? H : (hw < 36L ? 1 : sh);

    DU *csum = new DU[C];
    for (U32 k = 0; k < C; k++) csum[k] = DU0;
    for (U32 i = 0; i < h; i++) {
        INFO("\n");
        for (U32 k = 0; k < C; k++) {
            for (U32 j = 0; j < w; j++) {
                U64 n = j + i * w;
                if (n >= hw) { INFO("  "); continue; }
                
                DU r0 = v[k + (j>0 ? n - 1 : n) * C];
                DU r1 = v[k + n * C];
                DU x0 = (r0 - mean) * scale;
                DU x1 = (((r0 + r1) * 0.5) - mean) * scale;

                INFO("%c%c", map(x0), map(x1));  /// double width
                csum[k] += r1;
            }
            INFO("|");
        }
    }
    if (h > 1) {
        INFO("\nΣΣ=");
        for (U32 k = 0; k < C; k++) INFO("%6.3f ", csum[k]);
    }
    INFO("\n");
    
    delete csum;
}

__HOST__ void
Tensor::show(bool dump) {
    const U32 N  = this->N(), H = this->H(), W = this->W(), C = this->C();
    const U64 hw = (U64)H * W;

    DU mean  = avg();
    DU scale = 0.5 / std();            /// P=95%
    for (U32 n = 0; n < N; n++) {
        DU *d = slice(n);
        if (dump || hw < 100) {
            INFO("\nn=%d", n);
            _dump(d, H, W, C);
        }
        if (hw > 36L) _view(d, H, W, C, mean, scale);
    }
    INFO("\n");
}

#endif // T4_DO_OBJ

} // namespace t4::mu
