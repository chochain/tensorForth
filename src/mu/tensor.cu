/** -*- c++ -*-
 * @file
 * @brief Tensor class - ranked tensor impmementation i.e. vector, matrix, tensor, ...
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tensor.h"

namespace t4::mu {

#if T4_DO_OBJ
///=======================================================================
/// static methods
///
__KERN__ void
k_matmul(
    DU *A, DU *B, DU *O,   /* O[H*W*C] = A[H*K*C] @ B[K*W*C] */
    t4_mm_opt opt,
    U32 K, U32 H, U32 W)
{
    const U32 j  = blockIdx.x * blockDim.x + threadIdx.x;  ///< W  2T  range
    const U32 i  = blockIdx.y * blockDim.y + threadIdx.y;  ///< H  65M range
    const U32 c  = blockIdx.z,  C = gridDim.z;             ///< C
    const U64 z0 = ((U64)W * i + j) * C + c;               ///< output matrix index
    
    if (i < H && j < W && c < C) {                         /// * TODO: tiled
        DU  *ax, *bx;
        U64 ai, bi;
        if (opt & MM_A_TXP) {                              /// * transpose A
            ax = &A[(U64)C * i + c]; ai = (U64)H * C;
            bx = &B[(U64)C * j + c]; bi = (U64)W * C;
        }
        else if (opt & MM_B_TXP) {                         /// * transpose B
            ax = &A[(U64)C * K * i + c]; ai = (U64)C;
            bx = &B[(U64)C * K * j + c]; bi = (U64)C;
        }
        else {                                             /// * no tranposition
            ax = &A[(U64)C * K * i + c]; ai = (U64)C;
            bx = &B[(U64)C * j + c];     bi = (U64)W * C;
        }
        DU2 acc = DU0;                                     /// * TODO: suffle sum
//      acc += ax[k * C] * bx[k * N * C];                  /// * 8.1 ms 1Kx1K
        for (U32 k = 0; k < K; k++, ax += ai, bx += bi) {
            acc += (*ax) * (*bx);                          /// * 6.2 ms 1Kx1K
        }
        if (opt & MM_INC) O[z0] += acc;                    /// * increment O
        else              O[z0] =  acc;                    /// * overwrite O
    }
}
#if 0 /* Claude code */
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ─── Core device function ────────────────────────────────────────────────────
// Can be called from any __global__ or __device__ function.
// A: M×K,  B: K×N,  C: M×N  (row-major, all in device memory)
__device__ void device_matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    // Shared memory tiles — sized at compile time via TILE_SIZE
    __shared__ float sA[T4_DIM_SZ][T4_DIM_SZ];
    __shared__ float sB[T4_DIM_SZ][T4_DIM_SZ];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Walk across the shared K dimension in TILE_SIZE-wide strips
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {

        // Cooperative load of A tile — guard for non-divisible K/M
        int aCol = t * TILE_SIZE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // Cooperative load of B tile — guard for non-divisible K/N
        int bRow = t * TILE_SIZE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();   // ← barrier 1: tile is ready

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();   // ← barrier 2: tile consumed before next load
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ─── Example kernel that calls it ────────────────────────────────────────────
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    device_matmul(A, B, C, M, N, K);
}

// ─── Host launcher ───────────────────────────────────────────────────────────
void launch_matmul(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// ─── General GEMM: C = alpha * A * B + beta * C ──────────────────────────────
// Supports float, double, or __half via template.
// A: M×K,  B: K×N,  C: M×N  (row-major, device memory)
template <typename T>
__device__ void device_gemm(
    const T* __restrict__ A,
    const T* __restrict__ B,
          T* __restrict__ C,
    int M, int N, int K,
    T alpha, T beta)
{
    __shared__ T sA[TILE_M][TILE_K];
    __shared__ T sB[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    T acc = static_cast<T>(0);

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {

        int aCol = t * TILE_K + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : static_cast<T>(0);

        int bRow = t * TILE_K + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : static_cast<T>(0);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        T c_old = (beta != static_cast<T>(0)) ? C[row * N + col] : static_cast<T>(0);
        C[row * N + col] = alpha * acc + beta * c_old;
    }
}

// ─── Transposed inputs variant: C = alpha * op(A) * op(B) + beta * C ─────────
// transA / transB: 'N' = no transpose, 'T' = transpose
template <typename T>
__device__ void device_gemm_ex(
    const T* __restrict__ A,
    const T* __restrict__ B,
          T* __restrict__ C,
    int M, int N, int K,
    T alpha, T beta,
    char transA, char transB)
{
    __shared__ T sA[TILE_M][TILE_K];
    __shared__ T sB[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    T acc = static_cast<T>(0);

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {

        // Load A tile — respect transposition
        int aCol = t * TILE_K + threadIdx.x;
        if (transA == 'T') {
            // A is K×M stored row-major; op(A) is M×K → read A[aCol][row]
            sA[threadIdx.y][threadIdx.x] =
                (row < M && aCol < K) ? A[aCol * M + row] : static_cast<T>(0);
        } else {
            sA[threadIdx.y][threadIdx.x] =
                (row < M && aCol < K) ? A[row * K + aCol] : static_cast<T>(0);
        }

        // Load B tile — respect transposition
        int bRow = t * TILE_K + threadIdx.y;
        if (transB == 'T') {
            // B is N×K stored row-major; op(B) is K×N → read B[col][bRow]
            sB[threadIdx.y][threadIdx.x] =
                (bRow < K && col < N) ? B[col * K + bRow] : static_cast<T>(0);
        } else {
            sB[threadIdx.y][threadIdx.x] =
                (bRow < K && col < N) ? B[bRow * N + col] : static_cast<T>(0);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
        T c_old = (beta != static_cast<T>(0)) ? C[row * N + col] : static_cast<T>(0);
        C[row * N + col] = alpha * acc + beta * c_old;
    }
}

// ─── Kernels ──────────────────────────────────────────────────────────────────

template <typename T>
__global__ void gemm_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K, T alpha, T beta)
{
    device_gemm<T>(A, B, C, M, N, K, alpha, beta);
}

template <typename T>
__global__ void gemm_ex_kernel(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    T alpha, T beta,
    char transA, char transB)
{
    device_gemm_ex<T>(A, B, C, M, N, K, alpha, beta, transA, transB);
}

// ─── Host launchers ───────────────────────────────────────────────────────────

// Standard GEMM: C = alpha * A * B + beta * C
template <typename T>
void launch_gemm(
    const T* d_A, const T* d_B, T* d_C,
    int M, int N, int K,
    T alpha = static_cast<T>(1),
    T beta  = static_cast<T>(0),
    cudaStream_t stream = 0)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    gemm_kernel<T><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

// Extended GEMM with transpose flags
template <typename T>
void launch_gemm_ex(
    const T* d_A, const T* d_B, T* d_C,
    int M, int N, int K,
    T alpha, T beta,
    char transA, char transB,
    cudaStream_t stream = 0)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    gemm_ex_kernel<T><<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
}

// ─── Explicit instantiations ──────────────────────────────────────────────────
template void launch_gemm<float> (const float*,  const float*,  float*,  int, int, int, float,  float,  cudaStream_t);
template void launch_gemm<double>(const double*, const double*, double*, int, int, int, double, double, cudaStream_t);

template void launch_gemm_ex<float> (const float*,  const float*,  float*,  int, int, int, float,  float,  char, char, cudaStream_t);
template void launch_gemm_ex<double>(const double*, const double*, double*, int, int, int, double, double, char, char, cudaStream_t);
#endif 
///
/// GEMM kernel (used CUDA dynamic parallelism)
///     O = alpha * A x B + beta * O
///     where A = HxKxC, B = KxWxC, O = HxWxC
///
__KERN__ void
k_gemm(
    DU *A, DU *B, DU *O,  /* O[H*W*C] = a * A[H*K*C] @ B[K*W*C] + b * O[H*W*C] */
    DU alpha, DU beta,
    U32 K, U32 H, U32 W)
{
    const U32 j = threadIdx.x + blockIdx.x * blockDim.x;   ///< W
    const U32 i = threadIdx.y + blockIdx.y * blockDim.y;   ///< H
    const U32 c = blockIdx.z, C = gridDim.z;               ///< channel deep
    const U64 WC= W * C;
    const U64 z0= ((U64)W * i + j) * C + c;                ///< output index

    if (i < H && j < W && c < C) {                         /// * TODO: tiled
        DU *ax = &A[(U64)C * K * i + c];
        DU *bx = &B[(U64)C * j + c];
        DU2 acc = DU0;                                     /// * TODO: suffle sum
        for (U32 k = 0; k < K; k++, ax += C, bx += WC) {
            acc += (*ax) * (*bx);
        }
        O[z0] = alpha * acc + beta * O[z0];                /// * scaling
    }
}
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
    O.fill(DU0);
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
    O.fill(DU0);
    FORK4(k_batchnvar, A.data, G.data, O.data, (U64)H*W);

    for (int i=0; i< O.numel; i++) {
        O.data[i] = SQRT(O.data[i] / NHW);
    }
    return O;
}
__HOST__ Tensor&
Tensor::mm(
    Tensor &A, Tensor &B, Tensor &O, t4_mm_opt opt) {
    U32 H  = opt & MM_A_TXP ? A.W() : A.H();
    U32 Ka = opt & MM_A_TXP ? A.H() : A.W();
    U32 W  = opt & MM_B_TXP ? B.H() : B.W();
    U32 Kb = opt & MM_B_TXP ? B.W() : B.H();
    U32 N  = B.N(), C = B.C();                     /// B, O common dimensions
    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("  tensor#mm Ka(%d)!=Kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    MM_DB("  tensor#matmul K=%d => NHWC=[%d,%d,%d,%d]\n", Ka, N, H, W, C);
    
    for (U32 n = 0; n < N; n++) {
        DU *da = A.data, *db = B.slice(n), *dx = O.slice(n);
        FORK3(k_matmul, H, W, C, da, db, dx, opt, Ka);
    }
    return O;
}
///
/// tensor GEMM C' = alpha * A x B + beta * C
///
__HOST__ Tensor&
Tensor::gemm(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
    U32 H = A.H(), W = B.W(), Ka = A.W(), Kb = B.H();
    U32 N = B.N(), C = B.C();
    if (Ka != Kb || N != O.N() || C != O.C()) {
        ERROR("  tensor#gemm ka(%d)!=kb(%d) or N, C diff\n", Ka, Kb);
        return O;
    }
    MM_DB("  tensor#gemm K=%d, a=%g, b=%g => NHWC=[%d,%d,%d,%d]\n",
          Ka, alpha, beta, N, H, W, C);

    for (U32 n = 0; n < N; n++) {
        DU *da = A.data, *db = B.slice(n), *dx = O.slice(n);
        FORK3(k_gemm, H, W, C, da, db, dx, alpha, beta, Ka);
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
    static DU z;                                    ///< shared static memory
    if (numel < T4_DIM_SZ) {                        /// * cheaper for small loop
        z = DU0;
        for (int i = 0; i < numel; i++) z += data[i];
    }
    else {
        FORK1(k_sum, numel, data, &z);
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
    static DU v;
    FORK1(k_nvar, numel, data, &v, avg());       /// * 8x straight loop

    v = numel ? SQRT(v) : DU0;
    SCALAR(v); return v;
}
__HOST__ DU
Tensor::norm() {
    static DU n;
    FORK1(k_nvar, numel, data, &n, DU0);

    DU v = numel ? SQRT(n) : DU0;
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
