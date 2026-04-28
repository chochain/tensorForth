/** -*- c++ -*-
 * @file
 * @brief Math/Blas utility functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#include "t4math.h"

namespace t4 {
namespace cg = cooperative_groups;

#if T4_DO_OBJ
///
/// collect sum per every thread sum per stride (T4_DIM_SQ)
///
__GPU__ float
d__stride_sum(float *src, long numel, long tid) {
    float v { 0.0f };
    for (long i=tid; i < numel; i+=blockDim.x * gridDim.x) {
        v += src[i];
    }
    return v;
};
///
/// collect sum per every thread sum per stride (T4_DIM_SQ)
///
__GPU__ float
d__stride_var(float* __restrict__ src, float avg, long numel, long tid) {
    float v { 0.0f };
    for (long i=tid; i < numel; i+=blockDim.x * gridDim.x) {
        v += (src[i] - avg) * (src[i] - avg);
    }
    return v;
};
///
/// shuffle sum 32 to 1
///
__GPU__ float
d__warp_sum(float v) {
    for (int k = 16; k > 0; k /= 2) {
        v += __shfl_down_sync(0xffffffff, v, k);
    }
    return v;
}
///
/// collect sum per every thread sum per stride (T4_DIM_SQ)
///
__GPU__ float
d__rollup_sum(float* __restrict__ smem) {
    ///
    /// sum up all warps
    ///
    float v { 0.0f };
    #pragma unroll
    for (int i = 0; i < (T4_DIM_SQ>>5); i++) {
        v += smem[i];
    }
    return v;
}

__KERN__ void
k_sum_gemini(float *src, float *sum, long numel) {
    float v = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop (handles any N)
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        v += src[i];
    }

    // Reduce within warp
    v = d__warp_sum(v);

    // One thread per warp writes to shared memory
    __shared__ float _sum[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) _sum[wid] = v;
    __syncthreads();

    // Final reduction in shared memory
    if (wid == 0) {
        v = (threadIdx.x < (blockDim.x / 32)) ? _sum[lane] : 0;
        v = d__warp_sum(v);
        // ONLY ONE ATOMIC PER BLOCK instead of one per thread
        if (lane == 0) {
            atomicAdd(sum, v);
        }
    }
    __syncthreads();
}

__KERN__ void
k_sum(float* __restrict__ src, float* __restrict__ sum, long numel) {             ///< sum of T4_DIM_SQ threads
    auto const g   { cg::this_thread_block() };                /// total threads
    auto const tid { g.thread_rank() };                        /// thread id
    
    float v { d__stride_sum(src, numel, tid) };                /// one sum per thread by stride
    v = d__warp_sum(v);                                        /// a reduced sum per warp
    
    __shared__ float _sum[T4_DIM_SQ >> 5];                     ///< shared to keep warp sum
    auto const tpi { cg::tiled_partition<32>(g).thread_rank() };
    if (tpi == 0) _sum[tid >> 5] = v;                          /// collection from each warp
    g.sync();
    
    if (tid == 0) {
        *sum = d__rollup_sum(_sum);                            /// collection from each warp
    }
    g.sync();
}

__KERN__ void
k_nvar(float *src, float *avg, float var, long numel) {        ///< sum of T4_DIM_SQ threads
    __shared__ float _sum[T4_DIM_SQ >> 5];                     ///< shared to keep warp sum
    
    auto const g   { cg::this_thread_block() };                /// total threads
    auto const tid { g.thread_rank() };                        /// thread id

    float v { d__stride_var(src, var, numel, tid) };           /// collect one sum per thread
    v = d__warp_sum(v);                                        /// a reduced sum per warp
    
    auto const tpi { cg::tiled_partition<32>(g).thread_rank() };
    if (tpi == 0) _sum[tpi >> 5] = v;                        /// collection from each warp
    
    v = d__rollup_sum(_sum);                                   /// collection from each warp
    
    if (threadIdx.x == 0) *avg = v / numel;
}
///
///> Batch sum (NHW per channel)
///
__KERN__ void
k_batchsum(float *src, float *sum, long HW) {
    const long j  = (long)blockIdx.x*blockDim.x + threadIdx.x; ///< element index
    const int  c  = blockIdx.y, C = gridDim.y;                 ///< channel
    const int  n  = blockIdx.z;                                ///< batch slice index
    const long ns = HW * C * n;                                
    
    float v = (c < C && j < HW) ? src[ns + j * C + c] : 0.0f;
    v = d__warp_sum(v);                                        ///< collect sum per warp
    ///
    /// sum up atomically (per channel, for batchnorm)
    /// slower than grid-stride loop when blocks are many
    ///
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (c < C && tp.thread_rank() == 0) atomicAdd_block(&sum[c], v);  ///< serialize sum
}
///
///> batch variance (NHW per channel)
///
__KERN__ void
k_batchnvar(float *src, float*avg, float *var, long HW) {
    const long j  = (long)blockIdx.x * blockDim.x + threadIdx.x;  ///< element index
    const int  c  = blockIdx.y, C = gridDim.y;                    ///< channel
    const int  n  = blockIdx.z;                                   ///< batch slice index
    const long ns = HW * C * n;
    float v0 = (c < C && j < HW) ? src[(long)C * j + ns + c] - avg[c] : 0.0f;
    float v  = d__warp_sum(v0*v0);                                ///< collect sum per warp
    ///
    /// sum up atomically (per channel, for batchnorm)
    ///
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (c < C && tp.thread_rank() == 0) atomicAdd_block(&var[c], v);
}

struct Align4 { float data[4]; };
__KERN__ void
k_copy(float *src, float *dst, long n) {                      ///< Note: (src, dst)
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    // Use a struct to "hint" to the compiler we want 16-byte moves
    // while remaining safe for 4-byte alignment
    for (long i = idx * 4; i < n - 3; i += stride * 4) {
        *(Align4*)&dst[i] = *(Align4*)&src[i];
    }

    // Standard cleanup loop for the end
    for (long i = (n/4)*4 + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}
__KERN__ void
k_transpose(float *src, float *dst, int H, int W) {           ///< Note: (src, dst)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;      ///< W range 2G  * 1K = 2T,  U41
    const int i = blockIdx.y * blockDim.y + threadIdx.y;      ///< H range 65K * 1K = 65M, U26
    const int c = blockIdx.z, C = gridDim.z;                  ///< channel deep

    if (i < H && j < W && c < C) {
        dst[((long)H * j + i) * C + c] = src[((long)W * i + j) * C + c];
    }
}
__KERN__ void
k_identity(float *t, int H, int W) {                          ///< identity matrix (tensor)
    const float i01[2] = { 0.0f, 1.0f };
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z, C = gridDim.z;                  ///< channel deep

    if (i < H && j < W && c < C) {
        t[((long)W * i + j) * C + c] = i01[i==j];
    }
}

#define DU_LNX   1.0e-12                                      /** log clamp */
__KERN__ void
k_math(math_op op, float *A, float v, long n) {               ///< self modifying ops
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < n; j += step) {
        float ak = A[j];                                      ///< cache value
        switch(op) {
        case ABS:   A[j] = ABS(ak);                   break;
        case NEG:   A[j] = NEG(ak);                   break;
        case EXP:   A[j] = EXP(ak);                   break;  /// * clamped
        case LN:    A[j] = LN(MAX(ak, DU_LNX));       break;  /// * clamped
        case LOG:   A[j] = LOG(MAX(ak, DU_LNX));      break;  /// * clamped
        case TANH:  A[j] = TANH(ak);                  break;
        case RELU:  A[j] = RELU(ak);                  break;
        case SIGM:  A[j] = SIGMOID(ak);               break;
        case SQRT:  A[j] = SQRT(MAX(ak, 0.0));        break;  /// * guarded
        case RCP:   A[j] = RCP(ak);                   break;  /// 1/x
        case SAT:   A[j] = SAT(ak);                   break;  /// [0.0..1.0]
        case FILL:  A[j] = v;                         break;
        case GFILL: A[j] = v * j / n;                 break;  /// gradient fill
        case SCALE: A[j] *= v;                        break;
        case POW:   A[j] = POW(ak, v);                break;  /// x^v
        case ADD:   A[j] += v;                        break;
        case SUB:   A[j] -= v;                        break;
        case MUL:   A[j] *= v;                        break;
        case DIV:   A[j] /= v;                        break;
        default: printf("k_math op=%d not supported\n", op);
        }
    }
}
///
/// tensor-tensor element-wise ops (grid-stride implementation)
///
__KERN__ void
k_tt_op(math_op op, float *A, float *B, float *O, long n) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < n; j += step) {
        switch (op) {                                         /// no divergence
        case ADD: O[j] = A[j] + B[j]; break;
        case SUB: O[j] = A[j] - B[j]; break;
        case MUL: O[j] = A[j] * B[j]; break;                  /// * convolution
        case DIV: O[j] = A[j] / B[j]; break;
        }
    }
}
///
/// tensor-scalar element-wise ops (grid-stride implementation)
///
__KERN__ void
k_ts_op(math_op op, float *A, float v, float *O, long n) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < n; j += step) {
        switch (op) {                                         /// no divergence
        case ADD: O[j] = A[j] + v; break;
        case SUB: O[j] = A[j] - v; break;
        case MUL: O[j] = A[j] * v; break;                     /// * convolution
        case DIV: O[j] = A[j] / v; break;
        }
    }
}
///
/// Binary Cross-Entropy (clamps output to >= -100)
///
#define DU_EPS   1.0e-6                                       /* epsilon */
__KERN__ void
k_bce(float *O, float *T, long n) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < n; j += step) {
//        O[i] = ABS(T[i]) < DU_EPS ? LN(DU1 - O[i] + DU_EPS) : LN(O[i] + DU_EPS);
        O[j] = T[j] * LN(O[j] + DU_EPS) + (1.0f - T[j]) * LN(1.0f - O[j] + DU_EPS);
    }
}
///
///> check Nan or Inf
///
__KERN__ void
k_nan_inf(float *src, int *cnt, long numel) {
    const long j  = (long)blockIdx.x*blockDim.x + threadIdx.x; ///< element index
    int v = j < numel && (isnan(src[j]) || isinf(src[j])) ? 1 : 0;
    v = d__warp_sum(v);
    
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (tp.thread_rank() == 0) atomicAdd_block(cnt, v);        ///< serialize sum
}
__KERN__ void
k_dummy() {}

///=======================================================================
/// GEMM methods
///
#define TILE 16
__KERN__ void
k_gemm(                      ///< O[M*N*C] = a * A[M*K*C] @ B[K*N*C] + b * O[M*N*C]
    F32_XP A, F32_XP B, F32_XP O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N)
{
    const int n0 = threadIdx.x + blockIdx.x * blockDim.x;   ///< W
    const int m0 = threadIdx.y + blockIdx.y * blockDim.y;   ///< H
    const int c  = blockIdx.z, C = gridDim.z;               ///< channel deep
    const long WC = N * C;
    const long z0 = ((long)N * m0 + n0) * C + c;            ///< output index

    if (m0 < M && n0 < N && c < C) {                        /// * TODO: tiled
        float  *ax  = &A[(long)C * K * m0 + c];
        float  *bx  = &B[(long)C * n0 + c];
        double acc  = 0.0f;                                 /// * TODO: suffle sum
        for (int k = 0; k < K; k++, ax += C, bx += WC) {
            acc += (*ax) * (*bx);
        }
        O[z0] = alpha * acc + beta * O[z0];                 /// * scaling
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
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N)
{
    const int tx = threadIdx.x, n0 = blockIdx.x * TILE + tx;   ///< output column  (W dim)
    const int ty = threadIdx.y, m0 = blockIdx.y * TILE + ty;   ///< output row     (H dim)
    const int c  = blockIdx.z,  C  = gridDim.z;

    // Shared memory tiles — one for A, one for B
    __shared__ float _sA[TILE][TILE], _sB[TILE][TILE];

    double acc = 0.0f;

    // Load tile of A: rows [blockIdx.y*TILE .. +TILE), cols [s*TILE .. +TILE)
    //     Thread (ty, tx) loads A[i_tile, k_tile, c]
    //     where i_tile = blockIdx.y*TILE + ty,  k_tile = s*TILE + tx
    // Load tile of B: rows [s*TILE .. +TILE), cols [blockIdx.x*TILE .. +TILE)
    //     Thread (ty, tx) loads B[k_tile, j_tile, c]
    //     where k_tile = s*TILE + ty,  j_tile = blockIdx.x*TILE + tx
    
    // Sweep over K in strips of TILE
    const int nstrip = (K + TILE - 1) / TILE;
    for (int s = 0; s < nstrip; s++) {
        const int k_a = s * TILE + tx, k_b = s * TILE + ty;   ///< k-index for this thread's A, B load
        _sA[ty][tx] = (m0 < M && k_a < K) ? A[((long)m0 * K + k_a) * C + c] : 0.0f;
        _sB[ty][tx] = (k_b < K && n0 < N) ? B[((long)k_b * N + n0) * C + c] : 0.0f;
        __syncthreads();

        // ---- Accumulate dot product over this tile's k-strip ----
        #pragma unroll
        for (int t = 0; t < TILE; t++) acc += _sA[ty][t] * _sB[t][tx];
        
        __syncthreads();   ///< guard: don't overwrite tiles before all threads finish
    }
    // ---- Write output ----
    if (m0 < M && n0 < N) {
        const long z0 = ((long)m0 * N + n0) * C + c;
        O[z0] = acc * alpha + O[z0] * beta;
    }
}

// ---------------------------------------------------------------------------
// k_gemm — register-tiled GEMM, channel-last (interleaved) layout
//
//   O[M,N,C] = alpha * op(A) @ op(B)  +  beta * O[M,N,C] + Bias[C]
//
//   op(A) = tA ? A[K,M,C] : A[M,K,C]
//   op(B) = tB ? B[N,K,C] : B[K,N,C]
//
//   O[H,W,C] = alpha * A[H,K,C] @ B[K,W,C]  +  beta * O[H,W,C]
//
// Memory layout (stride C between consecutive logical elements):
//   A normal      A[i,k,c]  →  A[ (i*K + k)*C + c ]
//   A transposed  A[k,i,c]  →  A[ (k*M + i)*C + c ]
//   B normal      B[k,j,c]  →  B[ (k*N + j)*C + c ]
//   B transposed  B[j,k,c]  →  B[ (j*K + k)*C + c ]
//
// Each thread block covers a (BM × BN) output tile for one channel c.
// Each thread computes a (TM × TN) register sub-tile within that block tile.
//
// Top-left corner of this block's (BM × BN) output tile
// Top-left corner of this thread's (TM × TN) register tile inside the block tile
//
// note: 64-register, 4 x SM issue
// ---------------------------------------------------------------------------
__KERN__ void
k_gemm_tile_claude(
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N) 
{
    const int tx = threadIdx.x, ty = threadIdx.y;  ///< [T4_DIM_SZ, T4_DIM_SZ]
    const int c  = blockIdx.z,  C  = gridDim.z;    ///< channels

    // -------------------------------------------------------------------------
    // Shared memory tiles
    // -------------------------------------------------------------------------
    __shared__ float _sA[BK][BM];               ///< [k-strip depth][output rows]
    __shared__ float _sB[BK][BN];               ///< [k-strip depth][output cols]

    // -------------------------------------------------------------------------
    // Register accumulators — TM × TN per thread, zero-initialised
    // -------------------------------------------------------------------------
    float acc[TM][TN] = {};

    // -------------------------------------------------------------------------
    // Cooperative tile loading
    //
    //   256 threads collaboratively load BM*BK = 1024 elements into _sA
    //   and BK*BN = 1024 elements into _sB.  Each thread loads 4 of each.
    //   A linear loader index maps each thread to a fixed set of elements.
    // -------------------------------------------------------------------------
    const int loader_id = ty * T4_DIM_SZ + tx;   ///< [0, 256)
    const int NA = (BM * BK) / T4_DIM_SQ;        ///< (64*16)/256 = 4
    const int NB = (BK * BN) / T4_DIM_SQ;        ///< (16*64)/256 = 4

    // -------------------------------------------------------------------------
    // Main K-strip loop
    // -------------------------------------------------------------------------
    const int nstrip = (K + BK - 1) / BK;
    for (int s = 0; s < nstrip; s++) {
        const int K_BASE = s * BK;
        // -- Load _sA ----------------------------------------------------------
        // Flatten _sA[BM][BK] and distribute across 256 loaders, 4 each.
        // Element flat maps to _sA[z/BK][z%BK].
        // Normal:     A[m0, k0, c]  →  (m0*K + k0)*C + c
        // Transposed: A[k0, m0, c]  →  (k0*M + m0)*C + c
        #pragma unroll
        for (int n = 0; n < NA; n++) {
            const int  z  = loader_id * NA + n;
            const int  ar = z / BK,               ac = z % BK;
            const int  m0 = blockIdx.y * BM + ar, k0 = K_BASE + ac;
            const long ai = tA ? ((long)k0 * M + m0) * C + c
                               : ((long)m0 * K + k0) * C + c;
            _sA[ac][ar] = (m0 < M && k0 < K) ? A[ai] : 0.0f;
        }
        // -- Load _sB ----------------------------------------------------------
        // Flatten _sB[BK][BN] and distribute across 256 loaders, 4 each.
        // Element flat maps to _sB[flat/BN][flat%BN].
        // Normal:     B[k0, n0, c]  →  (k0*N + n0)*C + c
        // Transposed: B[n0, k0, c]  →  (n0*K + k0)*C + c
        #pragma unroll
        for (int n = 0; n < NB; n++) {
            const int  z  = loader_id * NB + n;
            const int  br = z / BN,      bc = z % BN;
            const int  k0 = K_BASE + br, n0 = blockIdx.x * BN + bc;
            const long bi = tB ? ((long)n0 * K + k0) * C + c
                               : ((long)k0 * N + n0) * C + c;
            _sB[br][bc] = (k0 < K && n0 < N) ? B[bi] : 0.0f;
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
        for (int k = 0; k < BK; k++) {
            float rA[TM], rB[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++) rA[m] = _sA[k][ty * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; n++) rB[n] = _sB[k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++) acc[m][n] += rA[m] * rB[n];
        }
        __syncthreads();   ///< guard before next strip overwrites shared mem
    }

    // -------------------------------------------------------------------------
    // Write TM×TN results back to O (bounds-checked)
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int gm = blockIdx.y * BM + ty * TM + m;
        if (gm >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const int gn = blockIdx.x * BN + tx * TN + n;
            if (gn >= N) continue;
            const long z0 = ((long)gm * N + gn) * C + c;
            O[z0] = acc[m][n] * alpha + O[z0] * beta;
        }
    }
}
///
/// double buffering (105 register, 2 x SM issue)
///
__KERN__ void
k_gemm_tile_claude_x2(
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta, bool tA, bool tB, int K, int M, int N) 
{
    const int tx = threadIdx.x, ty = threadIdx.y;  ///< [T4_DIM_SZ, T4_DIM_SZ]
    const int c  = blockIdx.z,  C  = gridDim.z;    ///< channels

    // -------------------------------------------------------------------------
    // Shared memory tiles
    // -------------------------------------------------------------------------
    __shared__ float _sA[2][BK][BM];        ///< [k-strip depth][output rows]
    __shared__ float _sB[2][BK][BN];        ///< [k-strip depth][output cols]

    // -------------------------------------------------------------------------
    // Register accumulators — TM × TN per thread, zero-initialised
    // -------------------------------------------------------------------------
    float acc[TM][TN] = {};

    // -------------------------------------------------------------------------
    // Cooperative tile loading
    //
    //   256 threads collaboratively load BM*BK = 1024 elements into _sA
    //   and BK*BN = 1024 elements into _sB.  Each thread loads 4 of each.
    //   A linear loader index maps each thread to a fixed set of elements.
    // -------------------------------------------------------------------------
    const int loader_id = ty * T4_DIM_SZ + tx;   ///< [0, 256)
    const int NA = (BM * BK) / T4_DIM_SQ;        ///< (64*16)/256 = 4
    const int NB = (BK * BN) / T4_DIM_SQ;        ///< (16*64)/256 = 4

    // -------------------------------------------------------------------------
    // Main K-strip loop
    // -------------------------------------------------------------------------
    const int nstrip = (K + BK - 1) / BK;

    // -------------------------------------------------------------------------
    // Lambda-style inline loader — loads one strip s into buffer slot buf.
    // Spelled out as a macro to avoid register pressure from a device function.
    //
    // For _sA: flat index z → (ac=z%BK, ar=z/BK)
    //          global coords: m0 = blockIdx.y*BM + ar
    //                         k0 = K_BASE + ac
    //          normal:     A[(m0*K + k0)*C + c]
    //          transposed: A[(k0*M + m0)*C + c]
    //
    // For _sB: flat index z → (br=z/BN, bc=z%BN)
    //          global coords: k0 = K_BASE + br
    //                         n0 = blockIdx.x*BN + bc
    //          normal:     B[(k0*N + n0)*C + c]
    //          transposed: B[(n0*K + k0)*C + c]
    // -------------------------------------------------------------------------
#define LOAD_TILE(buf, s)                                               \
    {                                                                   \
        const int K_BASE = (s) * BK;                                    \
        _Pragma("unroll")                                               \
        for (int _n = 0; _n < NA; _n++) {                               \
            const int  z  = loader_id * NA + _n;                        \
            const int  ar = z / BK,               ac = z % BK;          \
            const int  m0 = blockIdx.y * BM + ar, k0 = K_BASE + ac;     \
            const bool ok = (m0 < M && k0 < K);                         \
            const long ai = tA ? ((long)k0 * M + m0) * C + c            \
                               : ((long)m0 * K + k0) * C + c;           \
            if (ok) __pipeline_memcpy_async(&_sA[buf][ac][ar], &A[ai], sizeof(float)); \
            else    _sA[buf][ac][ar] = 0.0f;                            \
        }                                                               \
        _Pragma("unroll")                                               \
        for (int _n = 0; _n < NB; _n++) {                               \
            const int  z  = loader_id * NB + _n;                        \
            const int  br = z / BN,      bc = z % BN;                   \
            const int  k0 = K_BASE + br, n0 = blockIdx.x*BN + bc;       \
            const int  ok = (k0 < K && n0 < N);                         \
            const long bi = tB ? ((long)n0 * K + k0) * C + c            \
                               : ((long)k0 * N + n0) * C + c;           \
            if (ok) __pipeline_memcpy_async(&_sB[buf][br][bc], &B[bi], sizeof(float)); \
            else    _sB[buf][br][bc] = 0.0f;                            \
        }                                                               \
        __pipeline_commit();                                            \
    }
    // -------------------------------------------------------------------------
    // Compute tile — reads from buffer slot buf, accumulates into acc[][]
    // -------------------------------------------------------------------------
#define COMPUTE_TILE(buf)                                               \
    {                                                                   \
        _Pragma("unroll")                                               \
        for (int k = 0; k < BK; k++) {                                  \
            float rA[TM], rB[TN];                                       \
            _Pragma("unroll")                                           \
            for (int m = 0; m < TM; m++) rA[m] = _sA[(buf)][k][ty*TM+m];\
            _Pragma("unroll")                                           \
            for (int n = 0; n < TN; n++) rB[n] = _sB[(buf)][k][tx*TN+n];\
            _Pragma("unroll")                                           \
            for (int m = 0; m < TM; m++)                                \
                _Pragma("unroll")                                       \
                for (int n = 0; n < TN; n++) acc[m][n] += rA[m]*rB[n];  \
        }                                                               \
    }

    // -------------------------------------------------------------------------
    // Pipeline:
    //   1. preload strip 0 into buf 0
    //   2. for each subsequent strip:
    //        load strip s+1 into buf (s+1)%2    <- overlaps with...
    //        compute strip s from buf s%2        <- ...this
    //        sync to ensure load is done before next compute
    //   3. compute final strip from buf (nstrip-1)%2
    // -------------------------------------------------------------------------
    LOAD_TILE(0, 0);
    __pipeline_wait_prior(0);
    __syncthreads();                     // buf 0 ready before first compute
    
    for (int s = 0; s < nstrip - 1; s++) {
        // load next strip into the other buffer — no sync yet,
        // these global loads can overlap with the compute below
        LOAD_TILE((s + 1) % 2, s + 1);
        
        // compute current strip from the buffer we already synced on
        COMPUTE_TILE(s % 2);

        // now ensure the load we just issued is visible to all threads
        // before the next iteration's COMPUTE_TILE reads from it
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // compute the last strip — no more loads needed
    COMPUTE_TILE((nstrip - 1) % 2);
    __syncthreads();                     // guard before smem goes out of scope
    
#undef LOAD_TILE
#undef COMPUTE_TILE
    
    // -------------------------------------------------------------------------
    // Write TM×TN results back to O (bounds-checked)
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int gm = blockIdx.y * BM + ty * TM + m;
        if (gm >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const int gn = blockIdx.x * BN + tx * TN + n;
            if (gn >= N) continue;
            const long z0 = ((long)gm * N + gn) * C + c;
            O[z0] = acc[m][n] * alpha + O[z0] * beta;
        }
    }
}

#endif // T4_DO_OBJ
} // namespace t4

