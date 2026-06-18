/** -*- c++ -*-
 * @file
 * @brief Neural Network kernel modules
 * @note template file nmath.tcu is included at the tail of this file
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <float.h>              // -FLT_MAX
#include "nmath.h"
///
/// General Notations
///   xx1  : 1 stands for input,  i.g. i1 - x dimension index for input
///   xx0  : 0 stands for output, i.g. C0 - number of channels for output
///   n, N : index in mini-batch
///   c, C : index among channels
///   i, W : index in x dimension
///   j, H : index in y dimension
///   k    : general counter
///   z, HW: page index
/// 
namespace t4::nn {

#if (T4_DO_OBJ && T4_DO_NN)
// ---------------------------------------------------------------------------
// k_bias — add bias vector B[E0] to each row of O[N, E0], channel-last C=1
// ---------------------------------------------------------------------------
__KERN__ void k_bias(
    DP_R B, DP_W O, int N, int E0) {
    const int e0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n  = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e0 < E0 && n < N) {
        O[n * E0 + e0] += B[e0];
    }
}

__KERN__ void k_activate(
    t4_layer op,                           ///< function to call
    DP_R I, DP_W O, DP_W F,                ///< input, filter, output tensors
    DU alpha, long numel                   ///< number of tensor elements
    ) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < numel; j += step) {
        DU i = I[j];                                       ///< use register
        switch (op) {
        case L_RELU: O[j] = i > DU0                        /// * 1|0
            ? (F[j]=DU1, i)
            : (F[j]=DU0);                           break;
        case L_TANH:
            O[j] = i = TANH(i);                            /// * [-1, 1)
            F[j] = DU1 - i * i;                     break; /// * (1 - tanh^2)
        case L_SIGMOID:
            O[j] = i = SIGMOID(i);
            F[j] = i * (DU1 - i);                   break; /// * sig*(1 - sig)
        case L_SELU: O[j] = i > DU0                        /// * selu
            ? (F[j] = SELU_L, i)
            : (F[j] = SELU_LA * _EXP(i)) - SELU_LA; break;
        case L_LEAKYRL: O[j] = i > DU0
            ? (F[j] = DU1, i)
            : (F[j] = alpha) * i;                   break;
        case L_ELU: O[j] = i > DU0
            ? (F[j] = DU1, i)
            : (F[j] = alpha * _EXP(i)) - alpha;     break;
        case L_DROPOUT: O[j] = F[j] > alpha                /// * 1|0
            ? (F[j]=DU1, i)
            : (F[j]=DU0);                           break;
        }
    }
}
///
/// k_softmax_small — one block per sample, C ≤ 256
///
__KERN__ void k_softmax_small(
    DP_R I, DP_W O, int C) {
    __shared__ DU _smem[32];              ///< shared: partial max then partial sum
    __shared__ DU _max, _sum;             ///< block-wide max, sum

    const int CX = T4_DIM_SQ / 32;
    const int c  = threadIdx.x;           ///< channel index
    const int n  = blockIdx.x;            ///< sample index
    
    DP_R s = &I[(long)n * C];             ///< slice for input n = blockIdx.x
    DP_W d = &O[(long)n * C];             ///< slice for output n = blockIdx.x

    DU mx = (c < C) ? s[c] : -FLT_MAX;    ///< init max (register)
    WARP_MAX(mx);

    const int warp_id = c / 32;           ///< collect one result per warp into shared memory
    const int lane    = c % 32;
    if (lane == 0) _smem[warp_id] = mx;
    __syncthreads();

    if (c < 32) {                         /// final reduce across warps — first warp only
        mx = (c < CX) ? _smem[c] : -FLT_MAX;
        WARP_MAX(mx);
        if (c == 0) _max = mx;            ///< broadcast max to all threads
    }
    __syncthreads();

    DU sm = DU0;                         ///< init sum (register)
    if (c < C) {
        sm   = _EXP(s[c] - _max);        /// * numerical stability: x - max
        d[c] = sm;                       /// * keep for final average
    }
    WARP_SUM(sm);                        /// * partial sum
    if (lane == 0) _smem[warp_id] = sm;
    __syncthreads();
    
    if (c < 32) {                        /// * warp sum
        sm = (c < CX) ? _smem[c] : DU0;
        WARP_SUM(sm);
        if (c == 0) _sum = sm;           /// * broadcast sum to all threads
    }
    __syncthreads();

    if (c < C) d[c] /= _sum;             /// * device each elements
}

__KERN__ void k_softmax(
    DP_R I, DP_W O, int C) {
    __shared__ DU _smem[32];             ///< 8 warp results, padded to 32
    __shared__ DU _max, _sum;

    const int CX   = T4_DIM_SQ / 32;     ///< number of warps = 8
    const int c    = threadIdx.x;
    const int n    = blockIdx.x;         ///< sample index  ← fix
    const int step = blockDim.x;         ///< stride = T4_DIM_SQ = 256
    const int lane = c % 32;
    const int warp = c / 32;

    DP_R s = &I[(long)n * C];
    DP_W d = &O[(long)n * C];

    DU mx = -FLT_MAX;                    ///< strip max
    for (int j = c; j < C; j += step) {
        mx = MAX(mx, s[j]);
    }
    WARP_MAX(mx);
    if (lane == 0) _smem[warp] = mx;
    __syncthreads();

    if (c < 32) {                        ///< full warp 0 — no divergence
        mx = (c < CX) ? _smem[c] : -FLT_MAX;
        WARP_MAX(mx);
        if (c == 0) _max = mx;
    }
    __syncthreads();

    for (int j = c; j < C; j += step) {  /// * calc each exp(v - _max)
        d[j] = _EXP(s[j] - _max);
    }
    __syncthreads();

    DU sm = DU0;                         ///< stride sum
    for (int j = c; j < C; j += step) sm += d[j];
    WARP_SUM(sm);
    if (lane == 0) _smem[warp] = sm;
    __syncthreads();

    if (c < 32) {                        ///< full warp 0 — no divergence
        sm = (c < CX) ? _smem[c] : DU0;
        WARP_SUM(sm);
        if (c == 0) _sum = sm;
    }
    __syncthreads();

    for (int j = c; j < C; j += step) d[j] /= _sum; /// * normalize
}
///
///> Batch norm statistics: mean and variance in a single kernel.
///  One block per (channel, batch) pair.
///  gridDim = (C, N, 1),  blockDim = (min(HW, 1024), 1, 1)
///  No atomics — each block fully owns its (c, n) slice.
///
/// <<<(C,N,1),(HW,1,1)>>>
__KERN__ void
k_batchnorm_1(
    DP_R I,             ///< input  [N, HW, C] (NHWC)
    DP_W avg,           ///< output mean  [C]  (accumulated over N outside)
    DP_W var,           ///< output M2    [C]  (accumulated over N outside)
    int  HW             ///< H * W
) {
    __shared__ DU s_sum[32];   ///< one slot per warp (max 32 warps)
    __shared__ DU s_sq[32];

    const int  tx = threadIdx.x;
    const int  c  = blockIdx.x, C = gridDim.x;     ///< channels
    const int  n  = blockIdx.y;                    ///< sample index
    const long ns = (long)HW * C * n;

    DU t_sum = DU0, t_sq = DU0;
    for (int j = tx; j < HW; j += blockDim.x) {    /// * grid-stride sum
        DU v  = I[ns + j * C + c];
        t_sum += v;
        t_sq  += v * v;
    }

    WARP_SUM(t_sum);                               /// S0=[0,32),S32=[32,64),...
    WARP_SUM(t_sq);

    const int lane = tx & 31, warp_id = tx >> 5;
    if (lane == 0) {                               /// * collect warp sum
        s_sum[warp_id] = t_sum;                    /// * keeps S0,S32,...
        s_sq [warp_id] = t_sq;
    }
    __syncthreads();

    const int nwarp = (blockDim.x + 31) >> 5;      ///< (HW) > 32
    if (warp_id == 0) {
        t_sum = (lane < nwarp) ? s_sum[lane] : DU0;
        t_sq  = (lane < nwarp) ? s_sq [lane] : DU0;
        
        WARP_SUM(t_sum);                           /// * S0+S32+...
        WARP_SUM(t_sq);

        if (lane == 0) {
            atomicAdd(&avg[c], t_sum);
            atomicAdd(&var[c], t_sq);
        }
    }
}
// <<<(((C+31)/32, max(32,C)>>> one thread per channel
__KERN__ void
k_batchnorm_2(
    DP_X avg, DP_X var, long NHW, int C
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= C) return;                           /// * guard

    DU b_avg = avg[c] / (DU)NHW;
    DU b_var = var[c] / (DU)NHW - b_avg * b_avg;
    
    avg[c] = b_avg;
    var[c] = DU1 / (_SQRT(MAX(b_var,DU0)) + DU_EPS);
}
// <<<((HW+255)/256,C,N),(256,1,1)>>>
__KERN__ void
k_batchnorm_3(
    DP_R I, DP_W O, DP_W XH,                ///< input, output, x_hat tensors
    DP_R W, DP_R B,                         /// * gamma, beta
    DP_R avg, DP_R rvar,                    /// * mean, 1.0/(stdvar + e)
    int  HW                                 ///< H0=H1, W0==W1 (C0==C1)
) {
    __shared__ DU s_avg, s_var, s_w, s_b;

    const int  tx = threadIdx.x;                           ///< [0,256)
    const int  j  = blockIdx.x * blockDim.x + tx;          ///< spatial [0,HW)
    const int  c  = blockIdx.y, C = gridDim.y;             ///< channel [0,C)
    const int  n  = blockIdx.z;                            ///< batch   [0,N)
    const long k  = ((long)HW * n + j) * C + c;
    
    // one thread loads the 4 scalars for this block's channel
    if (tx == 0) {
        s_avg = avg[c];  s_var = rvar[c];   ///< average, rvar per channel
        s_w   = W[c];    s_b   = B[c];
    }
    __syncthreads();

    if (j < HW) {
        O[k] = (XH[k] = (I[k] - s_avg) * s_var) * s_w + s_b;
    }
}
///==============================================================================
/// backprop
///==============================================================================
/// convolution filter derivatives
/// TODO: stride, dilation, [C1]NCHW filter
///
// ---------------------------------------------------------------------------
// k_dlinear_db — add O[N, E0] to each dB[E0]
// ---------------------------------------------------------------------------
__KERN__ void k_dlinear_db(
    DP_R O, DP_W DB, int N, int E0) {
    const int e0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n  = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e0 < E0 && n < N) atomicAdd(&DB[e0], O[n * E0 + e0]);
}
///
/// k_dbatchnorm_1 - reduce, fused reduction
///
/// Computes per-channel, per-batch-slice:
///   sum_dout[n,c]      = Σ_{j} dout[n,(1,j),c]
///   sum_dout_xhat[n,c] = Σ_{j} dout[n,(1,j),c] * xhat[n,(1,j),c]
///
/// Two-level reduction: warp → shared memory → one atomicAdd per block,
///
///<<<((HW+255)/256,C,N),(256,1,1)>>>
__KERN__ void k_dbatchnorm_1(
    DP_R D0, DP_R XH,                   ///< upstream gradient, saved x_hat
    DP_W sum_d0,                        ///< out: Σ dout        [N*C]
    DP_W sum_d0xh,                      ///< out: Σ dout*x_hat  [N*C]
    int HW                              ///< H*W spatial elements
    ) {
    const int  tx = threadIdx.x;
    const int  j  = blockIdx.x * blockDim.x + tx;  ///< [0,HW)
    const int  c  = blockIdx.y, C  = gridDim.y;
    const int  n  = blockIdx.z, nc = C * n + c;
    const long k  = ((long)HW * n + j) * C + c;    ///< [NHWC]

    DU v1 = (j < HW) ? D0[k]         : DU0;
    DU v2 = (j < HW) ? D0[k] * XH[k] : DU0;

    // --- level 1: warp reduce ---
    WARP_SUM(v1);                                  /// * S0=[0,32),S32=[32,64),...
    WARP_SUM(v2);

    // --- level 2: shared memory across warps in this block ---
    const int nwarp = (blockDim.x + 31) >> 5;      ///< ((HW+255)/256)/32 warps
    const int lane  = tx & 31, warp_id = tx >> 5;

    extern __shared__ DU smem[];   ///< smem[0,nwarps)=v1, smem[nwarps,2*nwarps)=v2
    DU *s1 = &smem[0];
    DU *s2 = &smem[nwarp];
    
    if (lane == 0) {                           
        s1[warp_id] = v1; s2[warp_id] = v2;        /// * collect S0,S32,...
    }
    __syncthreads();

    if (warp_id == 0) {
        v1 = (lane < nwarp) ? s1[lane] : DU0;
        v2 = (lane < nwarp) ? s2[lane] : DU0;
        
        WARP_SUM(v1);                              /// * S0+S32+...
        WARP_SUM(v2);
        
        if (lane == 0 && c < C) {
            atomicAdd(&sum_d0[nc],   v1);
            atomicAdd(&sum_d0xh[nc], v2);
        }
    }
}
///
/// k_dbatchnorm_2  —  per-(n,c) epilogue
///
/// Launched with <<<N, C>>> after the k_dbatchnorm_1 (reduction) is complete.
/// Accumulates dw/db and pre-scales the sums so k_dbatchnorm_3 can use them
/// directly without any CPU<->GPU synchronisation.
///
///   db[c]   += Σ_n mean(dout)          (beta gradient)
///   dw[c]   += Σ_n mean(dout * x_hat)  (gamma gradient)
///   s1[n,c]  = mean_dout[n,c]          (used in dX)
///   s2[n,c]  = mean_dout_xhat[n,c]     (used in dX)
///
/// One thread per (n,c): no serial loop over N, but db/dw accumulation
/// across n now requires atomicAdd since each n is a separate block.
///
///<<<N, C>>>
__KERN__ void k_dbatchnorm_2(
    DP_W DW, DP_W DB,                   ///< d_gamma, d_beta accumulators [C]
    DP_X sum_d0,                        ///< in: Σ dout   [N*C] → mean_dout
    DP_X sum_d0xh,                      ///< in: Σ dout*x̂ [N*C] → mean_dout_xhat
    long NHW,                           ///< N*H*W
    bool do_train
    ) {
    const int c     = threadIdx.x, C = blockDim.x;    ///< channel index
    const int n     = blockIdx.x;                     ///< batch index
    const int nc    = n * C + c;
    
    const DU g_d0   = sum_d0[nc]   / (DU)NHW;
    const DU g_d0xh = sum_d0xh[nc] / (DU)NHW;

    /// overwrite sums in place; k_dbatchnorm_3 will read them
    sum_d0[nc]   = g_d0;
    sum_d0xh[nc] = g_d0xh;

    if (do_train) {
        atomicAdd(&DB[c], g_d0);        ///< serialize N*C threads
        atomicAdd(&DW[c], g_d0xh);
    }
}
///
/// k_dbatchnorm_3 —  fused final dX update
///
/// dX[n,j,c] = g_var[c] *
///   (dout[n,1,j,c] - g_d0[n,c] - x_hat[n,1,j,c] * g_d0xh[n,c])
///
/// Reads dout and xhat exactly once
/// Writes result directly into in.data (dX).
///
///<<<((HW+255)/256,C,N),(256,1,1)>>>
__KERN__ void k_dbatchnorm_3(
    DP_R W,                             ///< gamma  [C]
    DP_R D0,                            ///< upstream gradient      [N,H,W,C]
    DP_R XH,                            ///< saved x_hat            [N,H,W,C]
    DP_W DX,                            ///< output gradient tensor [N,H,W,C]
    DP_R g_d0,                          ///< mean(dout)             [N,C]
    DP_R g_d0xh,                        ///< mean(dout * x_hat)     [N,C]
    DP_R rvar,                          ///< 1/sqrt(var+e)          [C]
    int HW                              ///< H*W
    ) {
    const int  j  = blockIdx.x * blockDim.x + threadIdx.x;  ///< [0,HW)
    const int  c  = blockIdx.y, C  = gridDim.y;
    const int  n  = blockIdx.z, nc = C * n + c;
    const long k  = ((long)HW * n + j) * C + c;

    if (j < HW) {
        const DU g_var= rvar[c] * W[c];        ///< gamma * rvar
        DX[k] = g_var * (D0[k] - g_d0[nc] - XH[k] * g_d0xh[nc]);
    }
}

///==============================================================================
/// gradient
///==============================================================================
__KERN__ void k_sgd(
    DP_X G, DP_X DG, DP_X M,                   ///< w, dw, and momemtum tensors
    int N,                                     ///< batch size
    DU lr, DU b,                               ///< learn rate, beta(momemtum)
    long numel                                 ///< HWC
    ) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < numel; j += step) {
        DU dg = DG[j] / N;                                ///< dG batch avg
        if (ZEQ(b)) G[j] -= lr * dg;
        else {
            float mi = M[j] = b * M[j] + (DU1 - b) * dg;  ///< momentum
            G[j] -= lr * mi;                              /// * update gradient
        }
        DG[j] = DU0;                                      /// * zero after batch
    }
}

__KERN__ void k_adam(
    DP_X G, DP_X DG, DP_X M, DP_X V,           ///< w, dw, and momemtum tensors
    int N,                                     ///< batch size
    DU lrc, DU b1, DU b2,                      ///< corrected learn rate, beta(momemtum)
    long numel                                 ///< HWC
    ) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < numel; j += step) {
        const DU dg = DG[j];                              ///< dG (no batch avg)
        const DU mi = M[j] = b1 * M[j] + (DU1 - b1) * dg;        ///< momentum
        const DU vi = V[j] = b2 * V[j] + (DU1 - b2) * dg * dg;   ///< velocity
        
        G[j] -= lrc * mi / (_SQRT(vi) + DU_EPS);          /// * update gradient, clipped
        DG[j] = DU0;                                      /// * zero out dG for next round
    }
}

__KERN__ void k_adamw(
    DP_X G, DP_X DG, DP_X M, DP_X V,           ///< w, dw, and momemtum tensors
    int N,                                     ///< batch size
    DU lrc, DU b1, DU b2, DU wd,               ///< corrected learn rate, beta(momemtum)
    long numel                                 ///< HWC
    ) {
    const long tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const long step = gridDim.x * blockDim.x;
    for (long j = tx; j < numel; j += step) {
        const DU dg = DG[j];
        const DU mi = M[j] = b1 * M[j] + (DU1 - b1) * dg;
        const DU vi = V[j] = b2 * V[j] + (DU1 - b2) * dg * dg;

        G[j] -= lrc * (mi / (_SQRT(vi) + DU_EPS) - wd * dg); // ← decoupled weight decay
        DG[j] = DU0;
    }
}
#endif  // (T4_DO_OBJ && T4_DO_NN)

} // namespace t4::nn
//==========================================================================
