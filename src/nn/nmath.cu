/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <float.h>
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
///
/// convolution filter
/// Note: half-padding, no-stride 
/// TODO: stride, dilation, [C1]NCHW filter,
///       [see](https://github.com/vdumoulin/conv_arithmetic)
///
///==============================================================================
/// forward
///==============================================================================
template<int TS, int R>                    ///> tile size, kernel radius
__KERN__ void k_conv2d(
    DP_R I, DP_W O,                        ///> input I[HxW], output O[HxW]
    DP_R F, DP_R B,                        ///> kernel F[KxK], bias B[C]
    int H, int W, int C1, int C0           ///< (H0==H1, W0==W1), input/output channels
    ) {
    constexpr  int KS  = 2 * R + 1;                ///< center pixel + radius
    constexpr  int SSZ = (TS + KS - 1);
    
    __shared__ DU _I[SSZ][SSZ];                    ///< shared memory
    __shared__ DU _F[KS][KS];                      ///< cached filter, shared by all threads
    
    const int c0   = blockIdx.z % C0;              ///< ← c0 now from grid
    const int c1   = (blockIdx.z / C0) % C1;
    const int n    = blockIdx.z / (C0 * C1);
    const int HWC1 = H * W * C1, HWC0 = H * W * C0;

    DP_R n_I = I + n * HWC1;                       ///< input  slice [n]
    DP_W n_O = O + n * HWC0;                       ///< output slice [n]
    
    const int  tx = threadIdx.x, j0 = tx + blockIdx.x * TS;   ///< output coordinates
    const int  ty = threadIdx.y, i0 = ty + blockIdx.y * TS;   /// * i0,j0=0:15
    const long z0 = ((long)W * i0 + j0) * C0 + c0;    ///< output array index
    const int  load_id = ty * TS + tx;
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const long zf = (long)C0 * KS * KS * c1 + c0;     ///< filter index [C1,KS,KS,C0]
    if (tx < KS && ty < KS) {
        _F[ty][tx] = F[zf + (ty * KS + tx) * C0 + c0];///< base: filter[c1, 0, 0, c0]
    }
    ///
    /// load input tile _I — cooperative load, all TS×TS threads participate
    ///
    for (int t=load_id; t < SSZ*SSZ; t += TS*TS) {
        int si = t / SSZ, sj = t % SSZ;
        int gi = (blockIdx.y * TS) - R + si;
        int gj = (blockIdx.x * TS) - R + sj;
        _I[si][sj] = (gi >=0 && gi < H && gj >=0 && gj < W)
            ? n_I[((long)W * gi + gj) * C1 + c1] : DU0;       /// * cache input data
    }
    __syncthreads();                                 /// * smem write barrier
    ///
    /// Y = sum(W * X) + B
    /// accumulate in register, single global write at end — no atomicAdd
    /// each (n, i0, j0, c0) is owned by exactly one block → no race
    ///
    if (tx < TS && ty < TS && i0 < H && j0 < W) {    /// * each tile[14x14]
        DU sum = B[c0];                              ///< Y = B  (register)
        #pragma unroll
        for (int y = 0; y < KS; y++) {               /// * process one KS * KS cell
            #pragma unroll
            for (int x = 0; x < KS; x++) {
                sum += _F[y][x] * _I[ty + y][tx + x];     ///< Y += F * X
            }
        }
        n_O[z0] = sum;                               ///< single write, no atomic
    }
}

template<int KS>
__KERN__ void k_pool(
    t4_layer op,
    DP_R I, DP_W O, int H, int W)
{
    const int  KSQ = KS * KS;
    const long HW  = (long)H * W;
    const long k0  = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (k0 >= HW) return;

    const int  j0  = k0 % W;                                ///< output col
    const int  c   = blockIdx.y, C = gridDim.y;
    const long ns  = HW * C * blockIdx.z;                   ///< output batch offset
    const long z0  = ns + k0 * C + c;                       ///< output index
    const long z1  = (ns + k0 * C) * KSQ + j0 * KS * C + c; ///< input tile, acc=0.9897
    const long RI  = (long)(W - 1) * KS * C;                ///< row advance after each KS cols

    DP_R ix = &I[z1];                                       ///< first elem for max/min
    DU   v  = (op == L_MAXPOOL || op == L_MINPOOL) ? *ix : DU0;

    /// op hoisted outside loop — no per-iteration branch
    switch (op) {
    case L_USAMPLE:
    case L_AVGPOOL:
        for (int y = 0; y < KS; y++, ix += RI)
            for (int x = 0; x < KS; x++, ix += C) v += *ix;
        v /= KSQ;
        break;
    case L_MAXPOOL:
        for (int y = 0; y < KS; y++, ix += RI)
            for (int x = 0; x < KS; x++, ix += C) v = MAX(*ix, v);
        break;
    case L_MINPOOL:
        for (int y = 0; y < KS; y++, ix += RI)
            for (int x = 0; x < KS; x++, ix += C) v = MIN(*ix, v);
        break;
    }
    O[z0] = v;
}

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
    DP_R I, DP_W F, DP_W O,                ///< input, filter, output tensors
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
__KERN__ void
k_batchnorm_stat(
    DP_R   src,           ///< input  [N, HW, C] (NHWC)
    DP_W   avg,           ///< output mean  [C]  (accumulated over N outside)
    DP_W   var,           ///< output M2    [C]  (accumulated over N outside)
    long   HW             ///< H * W
) {
    extern __shared__ DU smem[];   ///< [blockDim.x] for sum, reused for sumsq
    DU *s_sum = smem;
    DU *s_sq  = smem + blockDim.x;

    const int  c  = blockIdx.x;                              ///< channel
    const int  n  = blockIdx.y;                              ///< batch index
    const int  C  = gridDim.x;
    const long ns = HW * C * n;

    // --- grid-stride accumulation into registers ---
    DU t_sum = DU0, t_sq = DU0;
    for (long j = threadIdx.x; j < HW; j += blockDim.x) {
        DU v = src[ns + j * C + c];
        t_sum += v;
        t_sq  += v * v;
    }

    // --- store into shared memory ---
    s_sum[threadIdx.x] = t_sum;
    s_sq [threadIdx.x] = t_sq;
    __syncthreads();

    // --- tree reduction in shared memory ---
    for (int stride = blockDim.x >> 1; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq [threadIdx.x] += s_sq [threadIdx.x + stride];
        }
        __syncthreads();
    }

    // --- final warp reduction (no __syncthreads needed inside a warp) ---
    if (threadIdx.x < 32) {
        DU ws = s_sum[threadIdx.x];
        DU wv = s_sq [threadIdx.x];
        WARP_SUM(ws);
        WARP_SUM(wv);

        // one write per block, serialised across N slices with global atomic
        if (threadIdx.x == 0) {
            atomicAdd(&avg[c], ws);
            atomicAdd(&var[c], wv);
        }
    }
}

// gridDim = (C, 1, 1), one thread per channel
__KERN__ void
k_batchnorm_calc(
    DP_X avg, DP_X var, long NHW) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    DU b_avg = avg[c] / NHW;
    DU b_var = var[c] / NHW - b_avg * b_avg;
    
    avg[c] = b_avg;
    var[c] = DU1 / (_SQRT(b_var) + DU_EPS);
}

__KERN__ void k_batchnorm(
    DP_R I, DP_W O, DP_W X,                ///< input, filter, output tensors
    DP_R avg, DP_R rvar,                   ///< mean, 1.0/(stdvar + e)
    DP_R w, DP_R b,                        ///< gamma, beta
    long HW                                ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const long j = (long)blockIdx.x * blockDim.x + threadIdx.x; ///< element index
    const int  c = blockIdx.y, n = blockIdx.z, C = gridDim.y;   ///< channel deep, batch id
    const long k = (HW * n + j) * C + c;                        ///< output tensor index

    if (j < HW) {
        O[k] = (X[k] = (I[k] - avg[c]) * rvar[c]) * w[c] + b[c];
    }
}
///==============================================================================
/// backprop
///==============================================================================
/// convolution filter derivatives
/// TODO: stride, dilation, [C1]NCHW filter
///
template<int TS, int R>
__KERN__ void k_dconv2d(
    DP_R I, DP_R O,
    DP_W DX, DP_R F, DP_W DF, DP_R DB,
    int H, int W, int C0, int C1, bool train)
{
    constexpr int KS  = 2 * R + 1;
    constexpr int SSZ = TS + KS - 1;
    __shared__ DU _I[SSZ][SSZ], _O[SSZ][SSZ];           ///< shared I/O blocks
    __shared__ DU _df[KS][KS];                          ///< shared filter

    const int c0   = blockIdx.z % C0;                   ///< ← c0 now from grid
    const int c1   = (blockIdx.z / C0) % C1;
    const int n    = blockIdx.z / (C0 * C1);
    const int HWC1 = H * W * C1, HWC0 = H * W * C0;

    DP_R n_I  = I  + n * HWC1;                          ///< I/O tile pointers
    DP_W n_O  = O  + n * HWC0;
    DP_W n_DX = DX + n * HWC1;

    const int tx = threadIdx.x, j1 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i1 = ty + blockIdx.y * TS;
    const int load_id = ty * TS + tx;

    const int zi = blockIdx.y * TS - R, zj = blockIdx.x * TS - R;

    /// load _I tile — this c1 slice
    for (int t = load_id; t < SSZ*SSZ; t += TS*TS) {
        int si = t / SSZ, sj = t % SSZ;
        int gi = zi + si,  gj = zj + sj;
        _I[si][sj] = (gi>=0 && gi<H && gj>=0 && gj<W)
            ? n_I[((long)W * gi + gj) * C1 + c1] : DU0;
    }

    /// load _O tile — this c0 slice  (loaded once, no loop)
    for (int t = load_id; t < SSZ*SSZ; t += TS*TS) {
        int si = t / SSZ, sj = t % SSZ;
        int gi = zi + si,  gj = zj + sj;
        _O[si][sj] = (gi>=0 && gi<H && gj>=0 && gj<W)
            ? n_O[((long)W * gi + gj) * C0 + c0] : DU0;
    }
    if (tx < KS && ty < KS) _df[ty][tx] = DU0;
    __syncthreads();

    /// dB — one block per c0, warp reduce then atomic
    DU db = (train && tx < TS && ty < TS) ? _O[ty+R][tx+R] : DU0;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        db += __shfl_down_sync(0xffffffff, db, off);
    if ((ty * blockDim.x + tx) % 32 == 0) atomicAdd((DU*)&DB[c0], db);

    const long zf = (long)C0 * KS*KS * c1 + c0;

    /// dX and dF
    DU dx_acc = DU0;                                   ///< register accumulator
    for (int y = 0, y1 = KS-1; y < KS; y++, y1--) {
        for (int x = 0, x1 = KS-1; x < KS; x++, x1--) {
            DU dy  = (tx < TS && ty < TS) ? _O[ty+y][tx+x] : DU0;
            int fi = zf + (y1 * KS + x1) * C0;

            if (tx < TS && ty < TS) dx_acc += F[fi] * dy;
            if (!train) continue;

            DU df0 = dy * ((tx < TS && ty < TS) ? _I[ty+y][tx+x] : DU0);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                df0 += __shfl_down_sync(0xffffffff, df0, off);
            if ((ty * blockDim.x + tx) % 32 == 0)
                atomicAdd(&_df[y][x], df0);            /// * df keeps tile-sum
        }
    }
    __syncthreads();

    /// dX — now needs atomic since multiple c0 blocks write same DX element
    if (tx < TS && ty < TS && i1 < H && j1 < W) {
        atomicAdd(&n_DX[((long)W * i1 + j1) * C1 + c1], dx_acc);
    }
    /// dF — write accumulated _df to global
    if (train && tx < KS && ty < KS) {
        atomicAdd(&DF[zf + (ty * KS + tx) * C0], _df[ty][tx]);
    }
}

template<int KS>
__KERN__ void k_dpool(
    t4_layer op, DP_X I, DP_R O, int H, int W)
{
    const int KSQ = KS * KS;
    const long HW = (long)H * W;
    const long k0 = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (k0 >= HW) return;
    
    const int  j0 = k0 % W;                                ///< output col
    const int  c  = blockIdx.y, C  = gridDim.y;
    const long ns = HW * C * blockIdx.z;                   ///< output batch offset

    const long z0 = ns + k0 * C + c;                       ///< output (dY) index
    const long z1 = (ns + k0 * C) * KSQ + j0 * KS * C + c; ///< input tile, acc=0.9897
    const long RI = (long)(W - 1) * KS * C;                ///< row advance after KS cols

    DP_X ix = &I[z1];
    DU *t  = ix;                                           ///< argmax/argmin ptr
    DU  v  = (op != L_AVGPOOL) ? *ix : O[z0];              ///< init value

    /// op hoisted outside loop
    switch (op) {
    case L_AVGPOOL: v /= KSQ;
    case L_USAMPLE:
        /// distribute dY equally across all KS*KS input cells
        for (int y = 0; y < KS; y++, ix += RI)
            for (int x = 0; x < KS; x++, ix += C) *ix = v;
        break;
    case L_MAXPOOL:
        /// zero all, find argmax, write dY to argmax position
        for (int y = 0; y < KS; y++, ix += RI) {
            for (int x = 0; x < KS; x++, ix += C) {
                DU dx = *ix; *ix = DU0;
                if (dx > v) { v = dx; t = ix; }
            }
        }
        *t = O[z0];                                        ///< dY at argmax
        break;
    case L_MINPOOL:
        /// zero all, find argmin, write dY to argmin position
        for (int y = 0; y < KS; y++, ix += RI) {
            for (int x = 0; x < KS; x++, ix += C) {
                DU dx = *ix; *ix = DU0;
                if (dx < v) { v = dx; t = ix; }
            }
        }
        *t = O[z0];                                        ///< dY at argmin
        break;
    }
}

// ---------------------------------------------------------------------------
// k_dlinear_db — add O[N, E0] to each dB[E0]
// ---------------------------------------------------------------------------
__KERN__ void k_dlinear_db(
    DP_R O, DP_W DB, int N, int E0) {
    const int e0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n  = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e0 < E0 && n < N) atomicAdd(&DB[e0], O[n * E0 + e0]);
}

__KERN__ void k_dbatchnorm_1(
    DP_W I, DP_X O, DP_R X,                    ///< input, output, x_hat tensors
    DP_R sum, DP_R g_var,                      ///< sum(x_hat), gamma/(stdvar+e)
    long HW                                    ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const long j  = (long)threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int  c  = blockIdx.y, C = gridDim.y;                 ///< channel deep
    const int  n  = blockIdx.z, nc = C * n + c;                ///< batch_id, sum/var index
    const long ns = HW * C * n;                                ///< batch slice index
    const long k  = (long)C * j + ns + c;                      ///< output tensor index
    const DU   _N = DU1 / gridDim.z;                           ///< 1.0/HWN

    if (c < C && j < HW) {
        I[k] = (O[k] - sum[nc] * _N) * g_var[nc];               /// * dX = g_var * (dout - sum(dout) / N)
        O[k] *= X[k];                                           /// * dout * x_hat
    }
}

__KERN__ void k_dbatchnorm_2(
    DP_W I, DP_R X, DP_R sum,                  ///< input, x_hat
    long HW                                    ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const long j = (long)blockIdx.x * blockDim.x + threadIdx.x; ///< element index
    const int c  = blockIdx.y, C = gridDim.y;                   ///< channel deep
    const int n  = blockIdx.z, nc = C * n + c;                  ///< batch_id, sum index
    const long ns= HW * C * n;                                  ///< batch slice index
    const long k = (long)C * j + ns + c;                        ///< output tensor index

    if (c < C && j < HW) I[k] -= X[k] * sum[nc];
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
#endif  // (T4_DO_OBJ && T4_DO_NN)

} // namespace t4::nn
//==========================================================================
