/** -*- c++ -*-
 * @file
 * @brief Tensor class - ranked tensor impmementation i.e. vector, matrix, tensor, ...
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
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
    FORK1(k_ts_op, A.numel, op, A.data, v, O.data);
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
    FORK1(k_tt_op, A.numel, op, A.data, B.data, O.data);
    return O;
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
__HOST__ Tensor&
Tensor::mm(
    Tensor &A, Tensor &B, Tensor &O, bool inc, bool tA, bool tB) {
    return gemm(A, B, O, DU1, inc ? DU1 : DU0, tA, tB);
}

__HOST__ Tensor&
Tensor::linear(
    Tensor &A, Tensor &B, Tensor &O, int H, int W, int K, DU alpha, DU beta, bool tA, bool tB) {
    MM_DB("  tensor#linear a=%g, b=%g => O[%d,%d] = A[%d,%d]%s @ B[%d,%d]%s\n",
          alpha, beta, H, W, H, K, tA ? "^T" : "", K, W, tB ? "^T" : "");

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
    FORK1(k_copy, A.numel, A.data, O.data);
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
///
/// matrix inversion (Gauss-Jordan with Pivot)
/// Note: Gauss-Jordan elimination is expensive O(N^3)
/// TODO: CDP
///
__HOST__ Tensor&
Tensor::inverse(Tensor &A, Tensor &I) {
    U32 m = A.H(), n = A.W();
    MM_DB("  tensor#inverse [%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return I; }
    DU *da = A.data, *di = I.data;
    auto swap_rows = [da, di, n](U32 u, U32 z) {
        for (U32 k = 0; k < n; k++) {         ///> TODO: swap entire row
            DU ta = da[k + z * n], ti = di[k + z * n];
            da[k + z * n] = da[k + u * n]; da[k + u * n] = ta;
            di[k + z * n] = di[k + u * n]; di[k + u * n] = ti;
        }
    };
    auto find_max = [da, n](U32 z) {
        int u = z;
        for (U32 i = z + 1; i < n; i++) {    ///> TODO: CDP reduce
            if (ABS(da[z + i * n]) > ABS(da[z + u * n])) u = i;
        }
        if (ABS(da[z + u * n]) < DU_EPS) {
            ERROR("tensor#inverse sigular!\n");
            return -1;
        }
        return u;
    };
    auto diag = [da, di, n](U32 z) {
        DU r0 = da[z + z * n];
        for (U32 k = 0; k < n; k++) {
            U32 i = k + z * n;
            di[i] /= r0;
            da[i] /= r0;
        }};
    auto elim = [da, di, n](U32 z) {
        for (U32 i = 0; i < n; i++) {
            DU r1 = da[z + i * n];
            for (U32 k = 0; i!=z && k < n; k++) {
                di[k + i * n] -= r1 * di[k + z * n];
                da[k + i * n] -= r1 * da[k + z * n];
            }
        }};
    for (U32 z = 0; z < n; z++) {
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
__HOST__ Tensor&
Tensor::lu(Tensor &A) {
    U32 m = A.H(), n = A.W();
    MM_DB("  tensor#lu [%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return A; }

    DU *da = A.data;
    auto elim = [da, n](U32 z) {
        DU ra = da[z + z * n];
        if (fabs(ra) < DU_EPS) return;      /// * if 0 skip the row
        for (U32 y = z + 1; y < n; y++) {
            DU r1 = da[z + y * n] / ra;     /// * substitution
            for (U32 k = z; k < n; k++) {
                da[k + y * n] -= r1 * da[k + z * n];
            }
            da[z + y * n] = r1;             /// L stored in A to save space
        }
    };
    for (U32 z = 0; z < n; z++) {
        elim(z);               /// * eliminate variables in upper triangle
	}
    return A;
}
///
/// LU (preprocessed) matrix inversion
/// TODO: CDP
///
__HOST__ Tensor&
Tensor::lu_inverse(Tensor &LU) {
    U32 m = LU.H(), n = LU.W();
    MM_DB("  tensor#lu_inverse [%d,%d]\n", m, n);
    DU *dd = LU.data;
    auto forward = [dd, n](int z) {
        for (int y = z + 1; y < n; y++) {
            DU r1 = dd[z + y * n];
            for (int k = 0; k < z; k++) {               /// columns before
                dd[k + y * n] -= dd[k + z * n] * r1;
            }
            dd[z + y * n] = -r1;                        /// current z column
        }};
    auto backward = [dd, n](int z) {
        DU r0 = RCP(dd[z + z * n]);
        dd[z + z * n] = r0;                             /// diag
        for (int k = z + 1; k < n; k++) {               /// current z row
            dd[k + z * n] *= r0;
        }
        for (int y = 0; y < z; y++) {                   /// factorize rows above
            DU r1 = dd[z + y * n];
            dd[z + y *  n] = -r1 * r0;                  /// current z column
            for (int k = z + 1; k < n; k++) {           /// columns after
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
__HOST__ Tensor&
Tensor::plu(Tensor &A, Tensor &P, int *ns) {
    U32 m = A.H(), n = A.W();
    MM_DB("  tensor#plu [%d,%d]\n", m, n);
    if (m != n) { ERROR("square matrix?"); return A; }

    DU *da = A.data, *dp = P.data;
    *ns = 0;                                  ///> initialize flip sign
    auto swap_rows = [da, dp, n](U32 u, U32 z) {
        DU t = dp[z]; dp[z] = dp[u]; dp[u] = t;
        for (U32 k = z; k < n; k++) {         ///> TODO: swap entire row
            t = da[k + z * n];
            da[k + z * n] = da[k + u * n];
            da[k + u * n] = t;
        }
    };
    auto find_max = [da, n](U32 z) {
        int u = z;
        for (U32 i = z + 1; i < n; i++) {    ///> TODO: CDP reduce
            if (ABS(da[z + i * n]) > ABS(da[z + u * n])) u = i;
        }
        if (ABS(da[z + u * n]) < DU_EPS) {
            MM_DB("  tensor#lu sigular!\n");
            return -1;
        }
        return u;
    };
    auto elim = [da, n](U32 z) {
        DU ra = da[z + z * n];
        if (fabs(ra) < DU_EPS) return;       /// * if 0 skip the row
        for (U32 y = z + 1; y < n; y++) {
            DU r1 = da[z + y * n] / ra;      /// * substitution
            for (U32 k = z; k < n; k++) {
                da[k + y * n] -= r1 * da[k + z * n];
            }
            da[z + y * n] = r1;              /// L stored in A to save space
        }
    };
    for (U32 z = 0; z < m; z++) dp[z] = z;   /// init permutation vector
    for (U32 z = 0; z < n; z++) {
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
__HOST__ DU
Tensor::sum() {
    DU z = DU0;
    if (numel < T4_DIM_SZ) {                        /// * cheaper for small loop
        for (int i = 0; i < numel; i++) z += data[i];
    }
    else {
        FORK1(k_sum, numel, data, &_tmp);
        z = _tmp;
    }
    SCALAR(z); return z;
}
__HOST__ DU
Tensor::avg() {
    DU v = sum() / numel;
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::std() {
    FORK1(k_nvar, numel, data, &_tmp, avg());       /// * 8x straight loop

    DU v = numel ? SQRT(_tmp) : DU0;
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::norm() {
    FORK1(k_nvar, numel, data, &_tmp, DU0);

    DU v = numel ? SQRT(_tmp) : DU0;
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::max() {
    DU v = data[0];
    for (U64 i=1; i < numel; i++) {              ///> TODO: CDP prefix sum
        v = MAX(data[i], v);
    }
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::min() {
    DU v = data[0];
    for (U64 i=1; i < numel; i++) {              ///> TODO: CDP prefix sum
        v = MIN(data[i], v);
    }
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::dot(Tensor &B) {
    DU  acc = DU0;
    if (rank == 1 && B.rank == 1 && numel == B.numel) {
        for (U64 k=0; k < numel; k++) {          ///> TODO: kernel
            acc += data[k] * B.data[k];
        }
    }
    else ERROR("A.dot(B) dim? %ld != %ld)\n", numel, B.numel);
    SCALAR(acc); return acc;
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
        FORK1(k_bce, numel, data, tgt.data);
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
    static int cnt;
    cnt = 0;
    FORK1(k_nan_inf, numel, data, &cnt);
    return cnt;
}
///=======================================================================
/// linear algebra methods
///=======================================================================
/// matrix determinant
///
__HOST__ DU
Tensor::det() {
    U32 m = H(), n = W();
    MM_DB("  tensor#det [%d,%d]\n", m, n);

    DU v = DU1;
    for (U32 z = 0; z < m; z++) v *= data[z + z * n];

    SCALAR(v); return v;
}
///
/// matrix upper triangle
///
__HOST__ Tensor&
Tensor::triu() {
    U32 m = H(), n = W();
    MM_DB("  tensor#upper [%d,%d]\n", m, n);

    for (U32 z = 1; z < m; z++) {
        for (U32 k = 0; k < z; k++) {
            data[k + z * n] = DU0;
        }
    }
    return *this;
}
///
/// matrix lower triangle with diag filled with 1
///
__HOST__ Tensor&
Tensor::tril() {
    U32 m = H(), n = W();
    MM_DB("  tensor#lower [%d,%d]\n", m, n);

    for (U32 z = 0; z < m; z++) {
        data[z + z * n] = DU1;
        for (U32 k = z + 1; k < n; k++) {
            data[k + z * n] = DU0;
        }
    }
    return *this;
}
///=======================================================================
/// Tensor life-cycle ops
///
__HOST__ Tensor&
Tensor::reset(void *mem, U64 sz, t4_obj tt, t4_layer fn) {
    MM_DB("  tensor#reset(%p,%ld)\n", mem, sz);
    init(sz, tt, 1);                                   /// T4Base attributes

    const U64 GB   = 1L << 30;
    const U16 s[4] = { 1, 1, 1, 1 };
    const U32 h[4] = {
        (U32)(sz > GB ? (sz>>30) : sz),
        (U32)(sz > GB ? GB : 1L),
        1, 1
    };
    const Tensor *t[4]= { NULL, NULL, NULL, NULL };
    data    = (DU*)mem;
    grad_fn = fn;
    memcpy(stride, s, sizeof(s));
    memcpy(shape,  h, sizeof(h));
    memcpy(grad,   t, sizeof(t));
    memcpy(mtum,   t, sizeof(t));
    
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
    FORK1(k_math, numel, op, data, v);
    return *this;
}

__HOST__ Tensor&
Tensor::normalize(DU avg, DU std) {
    FORK1(k_ts_op, numel, SUB, data, avg, data);
    FORK1(k_ts_op, numel, DIV, data, std, data);
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
