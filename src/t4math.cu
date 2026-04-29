/** -*- c++ -*-
 * @file
 * @brief Math/Blas utility functions implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#include <float.h>             // FLT_MAX
#include "t4math.h"

namespace t4 {
namespace cg = cooperative_groups;

#if T4_DO_OBJ

#define WARP_REDUCE(v)                              \
    for (int off = 16; off > 0; off >>=1)           \
        v += __shfl_down_sync(0xffffffff, v, off)
///
/// collect sum per every thread sum per stride (T4_DIM_SQ)
///
__KERN__ void
k_sum(F32_RP src, F32_WP sum, long numel) {
    __shared__ float _sum[32];
    int t0 = threadIdx.x,      tx = blockIdx.x * blockDim.x + t0;
    int tj = threadIdx.x % 32, ti = threadIdx.x / 32;

    float v =  { 0.0f };                                       /// grid-stride loop (handles any N)
    for (int i = tx; i < numel; i += blockDim.x * gridDim.x) {
        v += src[i];
    }
    WARP_REDUCE(v);

    if (tj == 0) _sum[ti] = v;                                 /// collect one thread per warp
    __syncthreads();

    if (ti == 0) {                                             /// share mem reduction
        v = (t0 < (blockDim.x / 32)) ? _sum[tj] : 0.0f;
        WARP_REDUCE(v);
        if (tj == 0) atomicAdd(sum, v);                        /// * one add per block
    }
    __syncthreads();
}

__KERN__ void
k_nvar(F32_RP src, float avg, F32_WP var, long numel) {        ///< sum of T4_DIM_SQ threads
    __shared__ float _sum[32];
    int t0 = threadIdx.x,      tx = blockIdx.x * blockDim.x + t0;
    int tj = threadIdx.x % 32, ti = threadIdx.x / 32;

    float v =  { 0.0f };                                       ///< grid-stride loop (handles any N)
    for (int i = tx; i < numel; i += blockDim.x * gridDim.x) {
        float v0 = src[i] - avg;
        v += v0 * v0;
    }
    WARP_REDUCE(v);
    
    // One thread per warp writes to shared memory
    if (tj == 0) _sum[ti] = v;
    __syncthreads();

    if (ti == 0) {                                             /// share mem reduction
        v = (t0 < (blockDim.x / 32)) ? _sum[tj] : 0.0f;
        WARP_REDUCE(v);
        if (tj == 0) atomicAdd(var, v);                        /// * one add per block
    }
}

// ---------------------------------------------------------------------------
// d__max / d__min
//
//   CUDA provides atomicMax/Min only for integers.  For positive floats the
//   IEEE-754 bit pattern preserves magnitude order, so we can reinterpret
//   the float as a uint32 and use the integer atomic safely.
//   For negative floats the sign bit inverts the order, so we flip all bits
//   before comparing (two's-complement trick).
//
//   Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide (B.12)
// ---------------------------------------------------------------------------
__GPU__ __forceinline__ void
d__max(float *addr, float val, bool find_max) {
    uint32_t raw_bits = __float_as_uint(val);                ///< original, unmodified
    uint32_t new_bits = raw_bits;
    
    if (new_bits & 0x80000000u) new_bits = ~new_bits;        /// * flip for comparison only
    
    uint32_t assumed, cur = __float_as_uint(*addr);
    do {
        assumed = cur;
        uint32_t cmp = (assumed & 0x80000000u) ? ~assumed : assumed;
        if (find_max  && cmp >= new_bits) break;
        if (!find_max && cmp <= new_bits) break;
        cur = atomicCAS((uint32_t*)addr, assumed, raw_bits);  /// * write original bits
    } while (cur != assumed);
}

__KERN__ void
k_max(F32_RP src, F32_WP rst, bool find_max, long numel) {   ///< FORK(k_max, numel, ..)
    __shared__ float _smem[T4_DIM_SQ];
    
    const int  t0     = threadIdx.x;
    const long tx     = (long)blockIdx.x * blockDim.x + t0;
    const long stride = (long)gridDim.x  * blockDim.x;

    float mx = find_max ? -FLT_MAX : FLT_MAX;

    #pragma unroll 4
    for (long j = tx; j < numel; j += stride) {               ///< grid-stride pass — each thread own max
        float v = src[j];
        mx = find_max ? fmaxf(mx, v) : fminf(mx, v);
    }
    
    _smem[t0] = mx;                                           ///< block-level shared memory tree reduction
    __syncthreads();

    #pragma unroll
    for (int half = T4_DIM_SQ >> 1; half > 0; half >>= 1) {
        if (t0 < half) {
            _smem[t0] = find_max
                ? fmaxf(_smem[t0], _smem[t0 + half])
                : fminf(_smem[t0], _smem[t0 + half]);
        }
        __syncthreads();
    }
    if (t0 == 0) d__max(rst, _smem[0], find_max);
}
///
///> Batch sum (NHW per channel)
///
__KERN__ void
k_batchsum(F32_RP src, F32_WP sum, long HW) {
    const long j  = (long)blockIdx.x*blockDim.x + threadIdx.x; ///< element index
    const int  c  = blockIdx.y, C = gridDim.y;                 ///< channel
    const int  n  = blockIdx.z;                                ///< batch slice index
    const long ns = HW * C * n;                                
    
    float v = (c < C && j < HW) ? src[ns + j * C + c] : 0.0f;
    WARP_REDUCE(v);                                            ///< collect sum per warp
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
k_batchnvar(F32_RP src, F32_RP avg, F32_WP var, long HW) {
    const long j  = (long)blockIdx.x * blockDim.x + threadIdx.x;  ///< element index
    const int  c  = blockIdx.y, C = gridDim.y;                    ///< channel
    const int  n  = blockIdx.z;                                   ///< batch slice index
    const long ns = HW * C * n;
    float v0 = (c < C && j < HW) ? src[(long)C * j + ns + c] - avg[c] : 0.0f;
    float v  = v0 * v0;
    WARP_REDUCE(v);                                               ///< collect sum per warp
    ///
    /// sum up atomically (per channel, for batchnorm)
    ///
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (c < C && tp.thread_rank() == 0) atomicAdd_block(&var[c], v);
}

struct Align4 { float data[4]; };
__KERN__ void
k_copy(F32_RP src, F32_WP dst, long n) {                          ///< Note: (src, dst)
    long tx     = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    // Use a struct to "hint" to the compiler we want 16-byte moves
    // while remaining safe for 4-byte alignment
    for (long i = tx * 4; i < n - 3; i += stride * 4) {
        *(Align4*)&dst[i] = *(Align4*)&src[i];
    }

    // Standard cleanup loop for the end
    for (long i = (n/4)*4 + tx; i < n; i += stride) {
        dst[i] = src[i];
    }
}
__KERN__ void
k_transpose(F32_RP src, F32_WP dst, int H, int W) {           ///< Note: (src, dst)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;      ///< W range 2G  * 1K = 2T,  U41
    const int i = blockIdx.y * blockDim.y + threadIdx.y;      ///< H range 65K * 1K = 65M, U26
    const int c = blockIdx.z, C = gridDim.z;                  ///< channel deep

    if (i < H && j < W && c < C) {
        dst[((long)H * j + i) * C + c] = src[((long)W * i + j) * C + c];
    }
}
__KERN__ void
k_identity(F32_WP T, int H, int W) {                          ///< identity matrix (tensor)
    const float i01[2] = { 0.0f, 1.0f };
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z, C = gridDim.z;                  ///< channel deep

    if (i < H && j < W && c < C) {
        T[((long)W * i + j) * C + c] = i01[i==j];
    }
}

#define DU_LNX   1.0e-12                                      /** log clamp */
__KERN__ void
k_math(math_op op, F32_XP A, float v, long n) {               ///< self modifying ops
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
/// tensor-scalar element-wise ops (grid-stride implementation)
///
__KERN__ void
k_ts_op(math_op op, F32_XP A, float v, F32_XP O, long n) {
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
/// tensor-tensor element-wise ops (grid-stride implementation)
///
__KERN__ void
k_tt_op(math_op op, F32_RP A, F32_RP B, F32_WP O, long n) {
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
/// Binary Cross-Entropy (clamps output to >= -100)
///
#define DU_EPS   1.0e-6                                       /* epsilon */
__KERN__ void
k_bce(F32_RP T, F32_XP O, long n) {
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
k_nan_inf(F32_RP src, int *cnt, long numel) {
    const long j = (long)blockIdx.x*blockDim.x + threadIdx.x; ///< element index
    
    int v = j < numel && (isnan(src[j]) || isinf(src[j])) ? 1 : 0;
    WARP_REDUCE(v);
    
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (tp.thread_rank() == 0) atomicAdd_block(cnt, v);        ///< serialize sum
}
__KERN__ void
k_dummy() {}

// ---------------------------------------------------------------------------
// k_dot
//
//   Computes   O[n, c] = alpha * dot(A[n,:,c], B[n,:,c]) + beta * O[n,c]
//
//   Tensors are channel-last, rank-2 (H=1, W=K, C=C, N=N):
//     A[n, k, c]  →  A.slice(n)[ k*C + c ]
//     B[n, k, c]  →  B.slice(n)[ k*C + c ]
//     O[n, c]     →  O.slice(n)[ c ]        (scalar per channel per sample)
//
//   Grid:  blockIdx.x = c  (channel),  blockIdx.y = n  (batch sample)
//   Block: threadIdx.x in [0, T4_DIM_SZ)
// ---------------------------------------------------------------------------
#define VLEN  4                   ///< floats loaded per thread per unrolled step
__KERN__ void
k_dot(
    F32_RP A, F32_RP B, F32_XP O,
    float alpha, float beta, int K, int C)
{
    __shared__ float _smem[T4_DIM_SZ];
    const int tx     = threadIdx.x;;                  ///< [0, T4_DIM_SZ)
    const int stride = T4_DIM_SZ * VLEN;              ///< elements consumed per outer step
    const int c      = blockIdx.x;                    ///< channel index

    // -------------------------------------------------------------------------
    // Phase 1: strided partial accumulation into a register
    //
    //   Thread tx owns elements k = tx, tx+T4_DIM_SZ, tx+2*T4_DIM_SZ, ...
    //   The inner VLEN-unrolled loop amortises loop overhead and lets the
    //   compiler issue independent FMAs for better ILP.
    // -------------------------------------------------------------------------
    float acc = { 0.0f };

    int k = tx * VLEN;                               ///< stride index
    #pragma unroll 4
    for (; k + stride <= K; k += stride) {
        #pragma unroll
        for (int v = 0; v < VLEN; v++) {
            const int idx = (k + v) * C + c;
            acc += A[idx] * B[idx];
        }
    }
    // Tail: handle remainder when K is not a multiple of stride
    #pragma unroll
    for (int v = 0; v < VLEN; v++) {
        const int kv = k + v;
        if (kv < K) {
            const int idx = kv * C + c;
            acc += A[idx] * B[idx];
        }
    }
    // -------------------------------------------------------------------------
    // Phase 2: warp-level reduce via shared memory
    //
    //   T4_DIM_SZ == 32 == warp size, so __syncthreads() is technically
    //   redundant here, but kept for correctness if T4_DIM_SZ is ever
    //   changed to a value larger than 32.
    // -------------------------------------------------------------------------
    _smem[tx] = acc;
    __syncthreads();

    #pragma unroll
    for (int half = T4_DIM_SZ >> 1; half > 0; half >>= 1) {
        if (tx < half) _smem[tx] += _smem[tx + half];
        __syncthreads();
    }
    // -------------------------------------------------------------------------
    // Thread 0 writes the final scalar for this (n, c) pair
    // -------------------------------------------------------------------------
    if (tx == 0) O[c] = _smem[0] * alpha + O[c] * beta;
}
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

/// ===========================================================================
/// matrix inversion kernel helpers
/// ===========================================================================
///
/// k_find_pivot — argmax of |A[z,i]| for i=z..n-1, returns row index in d_pivot
///
__KERN__ void
k_find_pivot(const float *da, int *d_pivot, int z, int K) {
    __shared__ float _val[T4_DIM_SQ];
    __shared__ int   _idx[T4_DIM_SQ];

    const int tx = threadIdx.x;
    float val = -1.0f;
    int   idx = z;

    for (int j = z + tx; j < K; j += blockDim.x) {             /// grid-stride over given rows i = z .. n-1
        float v = ABS(da[z + j * K]);
        if (v > val) { val = v; idx = j; }
    }

    _val[tx] = val;
    _idx[tx] = idx;
    __syncthreads();

    /// tree reduction: keep the (val, idx) with larger val
    #pragma unroll
    for (int half = T4_DIM_SQ >> 1; half > 0; half >>= 1) {
        if (tx < half) {
            if (_val[tx + half] > _val[tx]) {
                _val[tx] = _val[tx + half];
                _idx[tx] = _idx[tx + half];
            }
        }
        __syncthreads();
    }
    
    if (tx == 0) d_pivot[0] = (_val[0] < DU_EPS) ? -1 : _idx[0];  /// * -1 singular
}

// ---------------------------------------------------------------------------
// k_swap_rows — swap row z and row u in both A and I
//   1 thread per column k, grid = (ceil(n/T4_DIM_SQ), 1)
// ---------------------------------------------------------------------------
__KERN__ void
k_swap_rows(float *da, float *di, int u, int z, int K) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= K) return;

    const int iz = tx + z * K;
    const int iu = tx + u * K;

    float ta = da[iz]; da[iz] = da[iu]; da[iu] = ta;
    if (di) { float ti = di[iz]; di[iz] = di[iu]; di[iu] = ti; }
}
// ---------------------------------------------------------------------------
// k_diag — normalise pivot row z by diagonal element A[z,z]
//   1 thread per column k
// ---------------------------------------------------------------------------
__KERN__ void
k_diag(float *da, float *di, int z, int K) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= K) return;

    const float r0 = da[z + z * K];               ///< pivot value (already != 0)
    const int   j  = tx + z * K;
    da[j] /= r0;
    di[j] /= r0;
}
// ---------------------------------------------------------------------------
// k_elim — eliminate column z from every row i != z
//   1 thread per row i, each thread sweeps all k columns
//
//   Equivalent to the CPU elim lambda:
//     for i != z:  row_i -= A[z,i] * row_z
// ---------------------------------------------------------------------------
__KERN__ void
k_elim(float *da, float *di, int z, int K) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= K || tx == z) return;

    const float r1 = da[z + tx * K];             ///< A[z, i]
    if (ABS(r1) < DU_EPS) return;                ///< row already zeroed — skip

    for (int k = 0; k < K; k++) {
        const int ki = k + tx * K, kz = k + z * K;
        da[ki] -= r1 * da[kz];
        di[ki] -= r1 * di[kz];
    }
}

// ===========================================================================
// Stage 1 — k_getrf : LU factorisation with partial pivoting
//
//   Outer loop (host, sequential over pivot columns z = 0..n-1):
//     1. k_find_pivot  — find row with largest |A[z,i]|  i>=z
//     2. k_swap_rows   — swap that row into position z
//     3. k_lu_col      — compute L[:,z] and Schur complement
//
//   k_lu_col  (1 thread per row i > z):
//     L[i,z] = A[i,z] / U[z,z]           stored in lower triangle of A
//     for k > z:  A[i,k] -= L[i,z]*U[z,k]   Schur complement update
//
//   After all z:  da holds packed L\U
//     da[k + i*n]  k <  i  →  L[i,k]  (multipliers, unit diag implicit)
//     da[k + i*n]  k >= i  →  U[i,k]  (upper triangle with diagonal)
// ===========================================================================
__KERN__ void
k_lu_col(float *da, int z, int K) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx <= z || tx >= K) return;                  ///< rows below pivot only
    
    const float pivot = da[z + z * K];               ///< U[z,z]
    const float lik   = da[z + tx * K] / pivot;      ///< multiplier L[i,z]
    da[z + tx * K]  = lik;                           ///< store in-place (lower tri)

    for (int k = z + 1; k < K; k++)                  ///< Schur complement
        da[k + tx * K] -= lik * da[k + z * K];
}

// ===========================================================================
// Stage 2 — k_getri : solve A·X = I  using packed L\U
//
//   Each column j of the identity I is an independent right-hand side.
//   k_fsub and k_bwd_sub each launch n threads — one per column j.
//   The inner loop (over i) is serial within each thread, but all n
//   columns run fully in parallel.
//
//   k_fsub  — forward substitution   L · y_j = P · e_j
//     Apply pivot permutation to column j, then solve unit lower triangular.
//     y[i] = b[i] - sum_{k<i} L[i,k] * y[k]      (L diagonal = 1, implicit)
//
//   k_bsub  — backward substitution  U · x_j = y_j
//     x[i] = (y[i] - sum_{k>i} U[i,k] * x[k]) / U[i,i]
// ===========================================================================
__KERN__ void
k_fsub(const float *lu, const int *d_piv, float *di, int K) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;  ///< column of di
    if (tx >= K) return;

    /// apply row permutation to column j of I
    for (int k = 0; k < K; k++) {
        int pk = d_piv[k];
        if (pk != k) {
            float t = di[tx + k * K]; di[tx + k * K] = di[tx + pk * K]; di[tx + pk * K] = t;
        }
    }
    /// forward substitution: unit lower triangular (diagonal = 1 not stored)
    for (int k = 1; k < K; k++) {                          ///< rows of di
        float s = di[tx + k * K];
        for (int j = 0; j < k; j++)                        ///< inner rows (lower triangle of row k)
            s -= lu[j + k * K] * di[tx + j * K];           /// * L[k,j] * y[j]
        di[tx + k * K] = s;
    }
}

__KERN__ void
k_bsub(const float *lu, float *di, int K) {
    const int tx = blockIdx.x * blockDim.x + threadIdx.x; ///< column of di
    if (tx >= K) return;

    /// Backward substitution: upper triangular with explicit diagonal
    for (int j = K - 1; j >= 0; j--) {                    ///< rows of di
        float s = di[tx + j * K];
        for (int k = j + 1; k < K; k++)                   ///< inner rows (lower triangle of row k)
            s -= lu[k + j * K] * di[tx + k * K];          /// * U[i,k] * x[k]
        di[tx + j * K] = s / lu[j + j * K];               /// * divided by U[j,j]
    }
}

__KERN__ void
k_det(const float *lu, float *d_logdet, int *d_sign, int K) {
    __shared__ float _acc[T4_DIM_SQ];
    __shared__ int   _sgn[T4_DIM_SQ];

    const int tx = threadIdx.x;
    float acc  = 0.0f;                                    ///< logsum (use log for stability)
    int   sign = 1;

    for (int j = tx; j < K; j += blockDim.x) {            ///< block-stride
        float u = lu[j + j * K];                          ///< U[j,j] on diag
        if (u < 0.0f) { sign = -sign; u = -u; }
        acc += LOG(u);
    }
    
    _acc[tx] = acc;
    _sgn[tx] = sign;
    __syncthreads();

    #pragma unroll
    for (int half = T4_DIM_SQ >> 1; half > 0; half >>= 1) {
        if (tx < half) {
            _acc[tx] += _acc[tx + half];
            _sgn[tx] *= _sgn[tx + half];
        }
        __syncthreads();
    }
    if (tx == 0) { *d_logdet = _acc[0]; *d_sign = _sgn[0]; }
}

#endif // T4_DO_OBJ
} // namespace t4

