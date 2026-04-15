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
    DU *A, DU *B, DU *O,   /* O[M*N*C] = A[M*K*C] @ B[K*N*C] */
    t4_mm_opt opt,
    U32 K, U32 M, U32 N)
{
    const U32 n0 = blockIdx.x * blockDim.x + threadIdx.x;  ///< W  2T  range
    const U32 m0 = blockIdx.y * blockDim.y + threadIdx.y;  ///< H  65M range
    const U32 c  = blockIdx.z,  C = gridDim.z;             ///< C
    const U64 z0 = ((U64)N * m0 + n0) * C + c;             ///< output matrix index
    
    if (m0 < M && n0 < N && c < C) {                       /// * TODO: tiled
        DU  *ax, *bx;
        U64 ai, bi;
        if (opt & MM_A_TXP) {                              /// * transpose A
            ax = &A[(U64)C * m0 + c]; ai = (U64)M * C;
            bx = &B[(U64)C * n0 + c]; bi = (U64)N * C;
        }
        else if (opt & MM_B_TXP) {                         /// * transpose B
            ax = &A[(U64)C * K * m0 + c]; ai = (U64)C;
            bx = &B[(U64)C * K * n0 + c]; bi = (U64)C;
        }
        else {                                             /// * no tranposition
            ax = &A[(U64)C * K * m0 + c]; ai = (U64)C;
            bx = &B[(U64)C * n0 + c];     bi = (U64)N * C;
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
#define TILE 16
///
/// A: M×K,  B: K×N,  O: M×N  (row-major, all in device memory)
__KERN__ void
k_matmul_claude(
    const DU* __restrict__ A,
    const DU* __restrict__ B,
    DU*       __restrict__ O,
    U32 K, U32 M, U32 N)
{
    const U32 tx = threadIdx.x, n0 = blockIdx.x * TILE + tx;
    const U32 ty = threadIdx.y, m0 = blockIdx.y * TILE + ty;
    const U32 c  = blockIdx.z,  C  = gridDim.z;

    // Shared memory tiles — sized at compile time via TILE
    __shared__ DU _sA[TILE][TILE], _sB[TILE][TILE];
    
    DU2 acc = DU0;

    // Load tile of A: rows [blockIdx.y*TILE .. +TILE), cols [s*TILE .. +TILE)
    //     Thread (ty, tx) loads A[i_tile, k_tile, c]
    //     where i_tile = blockIdx.y*TILE + ty,  k_tile = s*TILE + tx
    // Load tile of B: rows [s*TILE .. +TILE), cols [blockIdx.x*TILE .. +TILE)
    //     Thread (ty, tx) loads B[k_tile, j_tile, c]
    //     where k_tile = s*TILE + ty,  j_tile = blockIdx.x*TILE + tx
    
    // Walk across the shared K dimension in TILE-wide strips
    const int nstrip = (K + TILE -1) / TILE;
    for (int s = 0; s < nstrip; s++) {
        U32 k_a = s * TILE + tx, k_b = s * TILE + ty;        ///< k-index for this thread's A, B load
        _sA[ty][tx] = (m0 < M && k_a < K) ? A[((U64)m0 * K + k_a) * C + c] : DU0;
        _sB[ty][tx] = (k_b < K && n0 < N) ? B[((U64)k_b * N + n0) * C + c] : DU0;
        __syncthreads();   // ← barrier 1: tile is ready

        #pragma unroll
        for (int t = 0; t < TILE; ++t) acc += _sA[ty][t] * _sB[t][tx];

        __syncthreads();   // ← barrier 2: tile consumed before next load
    }

    if (m0 < M && n0 < N) {
        const U64 z0 = ((U64)m0 * N + n0) * C + c;
        O[z0] = acc;
    }
}

__KERN__ void
k_gemm(
    DU *A, DU *B, DU *O,  /* O[M*N*C] = a * A[M*K*C] @ B[K*N*C] + b * O[M*N*C] */
    DU alpha, DU beta,
    U32 K, U32 M, U32 N)
{
    const U32 n0 = threadIdx.x + blockIdx.x * blockDim.x;   ///< W
    const U32 m0 = threadIdx.y + blockIdx.y * blockDim.y;   ///< H
    const U32 c  = blockIdx.z, C = gridDim.z;               ///< channel deep
    const U64 WC = N * C;
    const U64 z0 = ((U64)N * m0 + n0) * C + c;              ///< output index

    if (m0 < M && n0 < N && c < C) {                        /// * TODO: tiled
        DU *ax = &A[(U64)C * K * m0 + c];
        DU *bx = &B[(U64)C * n0 + c];
        DU2 acc = DU0;                                     /// * TODO: suffle sum
        for (U32 k = 0; k < K; k++, ax += C, bx += WC) {
            acc += (*ax) * (*bx);
        }
        O[z0] = alpha * acc + beta * O[z0];                /// * scaling
    }
}
// ---------------------------------------------------------------------------
// k_gemm — tiled GEMM for channel-last (interleaved) layout
//
//   O[H,W,C] = alpha * A[H,K,C] @ B[K,W,C]  +  beta * O[H,W,C]
//
// Thread assignment (one thread = one output element per channel):
//   threadIdx.x / blockIdx.x  →  j  (W dimension)
//   threadIdx.y / blockIdx.y  →  i  (H dimension)
//   blockIdx.z                →  c  (channel, independent GEMM)
//
// Shared memory tiles:
//   _sA[TILE][TILE]  holds a TILE×TILE sub-block of A[:,k-strip,c]
//   _sB[TILE][TILE]  holds a TILE×TILE sub-block of B[k-strip,:,c]
//
// Memory layout (channel-last, stride C between consecutive k/j elements):
//   A[i,k,c]  →  A_ptr[ i*(K*C) + k*C + c ]   (c fixed per block)
//   B[k,j,c]  →  B_ptr[ k*(W*C) + j*C + c ]   (c fixed per block)
//   O[i,j,c]  →  O_ptr[ i*(W*C) + j*C + c ]   (c fixed per block)
// ---------------------------------------------------------------------------
__KERN__ void
k_gemm_claude(
    const DU * __restrict__ A,
    const DU * __restrict__ B,
    DU *O,
    DU alpha, DU beta,
    U32 K, U32 M, U32 N)
{
    const U32 tx = threadIdx.x, n0 = blockIdx.x * TILE + tx;   ///< output column  (W dim)
    const U32 ty = threadIdx.y, m0 = blockIdx.y * TILE + ty;   ///< output row     (H dim)
    const U32 c  = blockIdx.z,  C  = gridDim.z;

    // Shared memory tiles — one for A, one for B
    __shared__ DU _sA[TILE][TILE], _sB[TILE][TILE];

    DU2 acc = DU0;

    // Load tile of A: rows [blockIdx.y*TILE .. +TILE), cols [s*TILE .. +TILE)
    //     Thread (ty, tx) loads A[i_tile, k_tile, c]
    //     where i_tile = blockIdx.y*TILE + ty,  k_tile = s*TILE + tx
    // Load tile of B: rows [s*TILE .. +TILE), cols [blockIdx.x*TILE .. +TILE)
    //     Thread (ty, tx) loads B[k_tile, j_tile, c]
    //     where k_tile = s*TILE + ty,  j_tile = blockIdx.x*TILE + tx
    
    // Sweep over K in strips of TILE
    const U32 nstrip = (K + TILE - 1) / TILE;
    for (U32 s = 0; s < nstrip; s++) {
        const U32 k_a = s * TILE + tx, k_b = s * TILE + ty;   ///< k-index for this thread's A, B load
        _sA[ty][tx] = (m0 < M && k_a < K) ? A[((U64)m0 * K + k_a) * C + c] : DU0;
        _sB[ty][tx] = (k_b < K && n0 < N) ? B[((U64)k_b * N + n0) * C + c] : DU0;
        __syncthreads();

        // ---- Accumulate dot product over this tile's k-strip ----
        #pragma unroll
        for (U32 t = 0; t < TILE; t++) acc += _sA[ty][t] * _sB[t][tx];
        
        __syncthreads();   ///< guard: don't overwrite tiles before all threads finish
    }
    // ---- Write output ----
    if (m0 < M && n0 < N) {
        const U64 z0 = ((U64)m0 * N + n0) * C + c;
        O[z0] = acc * alpha + O[z0] * beta;
    }
}

// ---------------------------------------------------------------------------
// Tile parameters
//
//   BK          — strip depth along K loaded into shared memory each iteration
//   [BM, BN]    — output covered by one thread block
//   [TM, TN]    — output computed by ONE thread (register tile)
//
//   Thread block shape  = (BN/TN, BM/TM)  →  (16, 16) = 256 threads
//
//   Shared memory per block:
//     _sA[BM][BK] = 64*16 = 1024 floats = 4 KB
//     _sB[BK][BN] = 16*64 = 1024 floats = 4 KB
//     Total: 8 KB  (well within 48 KB)
//
//   Arithmetic intensity per strip (256 threads):
//     FMAs  : 256 threads * TM * TN * BK  = 256 * 4 * 4 * 16 = 65536
//     Loads : BM*BK + BK*BN = 2048 floats
//     → ~32 FMAs per float loaded  (vs 1 in plain tiled, 16 in 1-output-per-thread)
// ---------------------------------------------------------------------------
#define BK      16
#define BM      64
#define BN      64
#define TM       4
#define TN       4

#define THREADS_X  (BN / TN)          /** 16 — threads covering W dimension */
#define THREADS_Y  (BM / TM)          /** 16 — threads covering H dimension */
#define NTHREADS   (THREADS_X * THREADS_Y)

// ---------------------------------------------------------------------------
// FORK3T — grid over (ceil(W/BN), ceil(H/BM), C)
// ---------------------------------------------------------------------------
#define FORK3T(fn,h,w,c,...) {               \
    const dim3 _b(THREADS_X, THREADS_Y, 1);  \
    const dim3 _g(((w) + BN - 1) / BN,       \
                  ((h) + BM - 1) / BM, c);   \
    fn<<<_g,_b>>>(__VA_ARGS__,h,w);          \
}

__KERN__ void
k_gemm_tile_gemini(
    DU *__restrict__ A,
    DU *__restrict__ B,
    DU *O,
    DU alpha, DU beta,
    U32 K, U32 M, U32 N) {
    /// Shared Memory Allocation
    __shared__ DU _sA[BK][BM], _sB[BK][BN];   ///< _sB transposed to avoid bank conflicts

    /// Register Accumulators (The 4x4 micro-tile, zero initialized)
    DU acc[TM][TN] = { DU0 };

    /// Global Row/Col for this thread block
    const U32 tx = threadIdx.x, bx = blockIdx.x * BN;
    const U32 ty = threadIdx.y, by = blockIdx.y * BM;
    const U32 c  = blockIdx.z,  C  = gridDim.z;

    /// Main K-Loop (Moving through the "inner" dimension)
    for (int k_off = 0; k_off < K; k_off += BK) {
        // Load TM rows of A per thread
        for (int m = 0; m < TM; m++) {
            U32 row = blockIdx.y * BM + ty * TM + m;
            U64 ai  = ((U64)row * K + (k_off + tx)) * C + c;
            _sA[tx][ty * TM + m] = (row < M && (k_off + tx) < K) ? A[ai] : DU0;
        }
        // Load TN cols of B per thread
        for (int n = 0; n < TN; n++) {
            U32 col = blockIdx.x * BN + tx * TN + n;
            U64 bi  = ((U64)(k_off + ty) * N + col) * C + c;
            _sB[ty][tx * TN + n] = (col < N && (k_off + ty) < K) ? B[bi] : DU0;
        }        
        __syncthreads();

        // 5. Inner Compute Loop (Shared Memory to Registers)
        for (int k = 0; k < BK; k++) {
            DU rA[TM], rB[TN];         ///< current row/col tile registers
            #pragma unroll
            for (int m = 0; m < TM; m++) rA[m] = _sA[k][ty * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; n++) rB[n] = _sB[k][tx * TN + n];

            // Perform 16 FMAs (Fused Multiply-Add) using only registers
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++)  acc[m][n] += rA[m] * rB[n];
            }
        }
        __syncthreads();
    }

    // 6. Write out results to Global Memory C
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            U32 row = blockIdx.y * BM + ty * TM + i;
            U32 col = blockIdx.x * BN + tx * TN + j;
            U64 z0  = (U64)row * N + col;
            if (row < M && col < N) O[z0] = alpha * acc[i][j] + beta * O[z0];
        }
    }
}

// ---------------------------------------------------------------------------
// k_gemm — register-tiled GEMM, channel-last (interleaved) layout
//
//   O[H,W,C] = alpha * A[H,K,C] @ B[K,W,C]  +  beta * O[H,W,C]
//
// Memory layout (stride C between consecutive logical elements):
//   A[i,k,c]  →  A[ i*(K*C) + k*C + c ]
//   B[k,j,c]  →  B[ k*(W*C) + j*C + c ]
//   O[i,j,c]  →  O[ i*(W*C) + j*C + c ]
//
// Each thread block covers a (BM × BN) output tile for one channel c.
// Each thread computes a (TM × TN) register sub-tile within that block tile.
// ---------------------------------------------------------------------------
__KERN__ void
k_gemm_tile_claude(
    DU * __restrict__ A,
    DU * __restrict__ B,
    DU *              O,
    DU alpha, DU beta,
    U32 K, U32 M, U32 N)
{
    const U32 tx = threadIdx.x;        ///< [0, THREADS_X) — col direction
    const U32 ty = threadIdx.y;        ///< [0, THREADS_Y) — row direction
    const U32 c  = blockIdx.z;         ///< channel index
    const U32 C  = gridDim.z;          ///< total channels

    // Top-left corner of this block's (BM × BN) output tile
//    const U32 by = blockIdx.y * BM;
//    const U32 bx = blockIdx.x * BN;

    // Top-left corner of this thread's (TM × TN) register tile inside the block tile
//    const U32 ry = ty * TM;            ///< row offset within block tile
//    const U32 rx = tx * TN;            ///< col offset within block tile

    // -------------------------------------------------------------------------
    // Shared memory tiles
    // -------------------------------------------------------------------------
    __shared__ DU _sA[BM][BK];         ///< [output rows][k-strip depth]
    __shared__ DU _sB[BK][BN];         ///< [k-strip depth][output cols]

    // -------------------------------------------------------------------------
    // Register accumulators — TM × TN per thread, zero-initialised
    // -------------------------------------------------------------------------
    DU2 acc[TM][TN] = { DU0 };

    // -------------------------------------------------------------------------
    // Cooperative tile loading
    //
    //   256 threads collaboratively load BM*BK = 1024 elements into _sA
    //   and BK*BN = 1024 elements into _sB.  Each thread loads 4 of each.
    //   A linear loader index maps each thread to a fixed set of elements.
    // -------------------------------------------------------------------------
    const U32 loader_id = ty * THREADS_X + tx;   ///< [0, 256)
    const U32 NA = (BM * BK) / NTHREADS;         ///< (64*16)/256 = 4
    const U32 NB = (BK * BN) / NTHREADS;         ///< (16*64)/256 = 4

    // -------------------------------------------------------------------------
    // Main K-strip loop
    // -------------------------------------------------------------------------
    const U32 nstrip = (K + BK - 1) / BK;

    for (U32 s = 0; s < nstrip; s++) {
        const U32 K_BASE = s * BK;
        // -- Load _sA ----------------------------------------------------------
        // Flatten _sA[BM][BK] and distribute across 256 loaders, 4 each.
        // Element flat maps to _sA[z/BK][z%BK].
        #pragma unroll
        for (U32 n = 0; n < NA; n++) {
            const U32 z  = loader_id * NA + n;
            const U32 ar = z / BK,               ac = z % BK;
            const U32 m0 = blockIdx.y * BM + ar, k0 = K_BASE + ac;
            const U64 ai = ((U64)m0 * K + k0) * C + c;
            _sA[ar][ac] = (m0 < M && k0 < K) ? A[ai] : DU0;
        }
        // -- Load _sB ----------------------------------------------------------
        // Flatten _sB[BK][BN] and distribute across 256 loaders, 4 each.
        // Element flat maps to _sB[flat/BN][flat%BN].
        #pragma unroll
        for (U32 n = 0; n < NB; n++) {
            const U32 z  = loader_id * NB + n;
            const U32 br = z / BN,      bc = z % BN;
            const U32 k0 = K_BASE + br, n0 = blockIdx.x * BN + bc;
            const U64 bi = ((U64)k0 * N + n0) * C + c;
            _sB[br][bc] = (k0 < K && n0 < N) ? B[bi] : DU0;
        }
        __syncthreads();   ///< all tiles loaded before any thread reads them

        // -- Register-tiled dot product ---------------------------------------
        // For each of the BK steps in this strip:
        //   1. Load TM values from As into rA[]  (one column of A sub-tile)
        //   2. Load TN values from Bs into b_reg[]  (one row   of B sub-tile)
        //   3. Outer-product: TM×TN FMAs, every operand lives in a register.
        //
        // This eliminates repeated shared-memory reads for the inner loop:
        // each rA / rB value is loaded once and reused TN / TM times.
        #pragma unroll
        for (U32 k = 0; k < BK; k++) {
            DU rA[TM], rB[TN];
            #pragma unroll
            for (U32 m = 0; m < TM; m++) rA[m] = _sA[tx * TM + m][k];
            #pragma unroll
            for (U32 n = 0; n < TN; n++) rB[n] = _sB[k][ty * TN + n];
            #pragma unroll
            for (U32 m = 0; m < TM; m++)
                #pragma unroll
                for (U32 n = 0; n < TN; n++) acc[m][n] += rA[m] * rB[n];
        }
        __syncthreads();   ///< guard before next strip overwrites shared mem
    }

    // -------------------------------------------------------------------------
    // Write TM×TN results back to O (bounds-checked)
    // -------------------------------------------------------------------------
    #pragma unroll
    for (U32 m = 0; m < TM; m++) {
        const U32 gm = blockIdx.y * BM + ty * TM + m;
        if (gm >= M) continue;
        #pragma unroll
        for (U32 n = 0; n < TN; n++) {
            const U32 gn = blockIdx.x * BN + tx * TN + n;
            if (gn >= N) continue;
            const U64 z0 = ((U64)gm * N + gn) * C + c;
            O[z0] = acc[m][n] * alpha + O[z0] * beta;
        }
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
Tensor::gemm2(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
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
        FORK3(k_gemm_claude, H, W, C, da, db, dx, alpha, beta, Ka);
    }
    return O;
}
__HOST__ Tensor&
Tensor::gemm3(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
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
        FORK3(k_gemm_tile_gemini, H, W, C, da, db, dx, alpha, beta, Ka);
    }
    return O;
}
__HOST__ Tensor&
Tensor::gemm4(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta) {
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
        FORK3T(k_gemm_tile_claude, H, W, C, da, db, dx, alpha, beta, Ka);
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
