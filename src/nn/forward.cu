/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <float.h>
#include "model.h"
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
#if (T4_DO_OBJ && T4_DO_NN)

namespace t4::nn {
///
/// convolution filter
/// Note: half-padding, no-stride 
/// TODO: stride, dilation, [C1]NCHW filter,
///       [see](https://github.com/vdumoulin/conv_arithmetic)
///
template<int TS, int R>            ///> tile size, kernel radius
__KERN__ void k_conv2d(
    DU *I, DU *F, DU *B, DU *O,    ///> input I[HxW], F[KxK] kernel, B[C] bias, output O[HxW]
    int H, int W, int C1, int C0   ///< (H0==H1, W0==W1), input/output channels
    ) {
    constexpr  int KS  = 2 * R + 1;                   ///< center pixel + radius
    constexpr  int SSZ = (TS + KS - 1);
    
    __shared__ float _I[SSZ][SSZ];                    ///< shared memory
    __shared__ float _F[KS][KS];                      ///< cached filter, shared by all threads
    
    const int c0   = blockIdx.z % C0;                   ///< ← c0 now from grid
    const int c1   = (blockIdx.z / C0) % C1;
    const int n    = blockIdx.z / (C0 * C1);
    const int HWC1 = H * W * C1, HWC0 = H * W * C0;

    DU *n_I  = I + n * HWC1;                          ///< input  slice [n]
    DU *n_O  = O + n * HWC0;                          ///< output slice [n]
    
    const int  tx = threadIdx.x, j0 = tx + blockIdx.x * TS;   ///< output coordinates
    const int  ty = threadIdx.y, i0 = ty + blockIdx.y * TS;   /// * i0,j0=0:15
    const long z0 = ((long)W * i0 + j0) * C0 + c0;    ///< output array index
    const int  load_id = ty * TS + tx;
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const long zf = (long)C0 * KS * KS * c1 + c0;       ///< filter index [C1,KS,KS,C0]
    if (tx < KS && ty < KS) {
        _F[ty][tx] = F[zf + (ty * KS + tx) * C0 + c0];  ///< base: filter[c1, 0, 0, c0]
    }
    ///
    /// load input tile _I — cooperative load, all TS×TS threads participate
    ///
    for (int t=load_id; t < SSZ*SSZ; t += TS*TS) {
        int si = t / SSZ, sj = t % SSZ;
        int gi = (blockIdx.y * TS) - R + si;
        int gj = (blockIdx.x * TS) - R + sj;
        _I[si][sj] = (gi >=0 && gi < H && gj >=0 && gj < W)
            ? n_I[((long)W * gi + gj) * C1 + c1] : DU0;               /// * cache input data
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
        n_O[z0] = sum;                                     ///< single write, no atomic
    }
}

// ---------------------------------------------------------------------------
// k_bias — add bias vector B[E0] to each row of O[N, E0], channel-last C=1
// ---------------------------------------------------------------------------
__KERN__ void k_bias(DU *B, DU *O, int N, int E0) {
    const int e0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n  = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e0 < E0 && n < N) {
        O[n * E0 + e0] += B[e0];
    }
}

#define KS_TILE(fn)                                 \
    for (int y = 0; y < KS; y++, ix += RI)          \
        for (int x = 0; x < KS; x++, ix += C) fn


template<int KS>
__KERN__ void k_pool(t4_layer op, DU *I, DU *O, U32 H, U32 W)
{
    const U32 KSQ = KS * KS;
    const U64 HW  = (U64)H * W;
    const U64 k0  = (U64)blockIdx.x * blockDim.x + threadIdx.x;

    if (k0 >= HW) return;

    const U32 j0  = k0 % W;                                ///< output col
    const U32 c   = blockIdx.y, C = gridDim.y;
    const U64 ns  = HW * C * blockIdx.z;                   ///< output batch offset
    const U64 z0  = ns + k0 * C + c;                       ///< output index
    const U64 z1  = (ns + k0 * C) * KSQ + j0 * KS * C + c; ///< input tile, acc=0.9897
    const U64 RI  = (U64)(W - 1) * KS * C;                 ///< row advance after each KS cols

    DU *ix = &I[z1];                                       ///< first elem for max/min
    DU  v  = (op == L_MAXPOOL || op == L_MINPOOL) ? DU0 : *ix;

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

#define SELU_L  1.0507                     /** Selu lambda */
#define SELU_LA 1.7581                     /** Selu alpha  */

__KERN__ void k_activate(
    t4_layer op, DU *I, DU *F, DU *O,      ///< func, input, filter, output tensors
    DU alpha, U64 numel                    ///< number of tensor elements
    ) {
    const U64 tx   = blockIdx.x * blockDim.x + threadIdx.x;
    const U64 step = gridDim.x * blockDim.x;
    for (U64 j = tx; j < numel; j += step) {
        DU i = I[j];                                       ///< use register
        switch (op) {
        case L_RELU: O[j] = i > DU0                        /// * 1|0
            ? (F[j]=DU1, i)
            : (F[j]=DU0);                          break;
        case L_TANH:
            O[j] = 0.5 * (DU1 + (i=TANH(i)));              /// * scaled to [0,1)
            F[j] = DU1 - i*i;                      break;  /// * (1 - tanh^2)
        case L_SIGMOID:
            O[j] = i = SIGMOID(i);
            F[j] = i * (DU1 - i);                  break;  /// * sig*(1 - sig)
        case L_SELU: O[j] = i > DU0                        /// * selu
            ? (F[j] = SELU_L, i)
            : (F[j] = SELU_LA * EXP(i)) - SELU_LA; break;
        case L_LEAKYRL: O[j] = i > DU0
            ? (F[j] = DU1, i)
            : (F[j] = alpha) * i;                  break;
        case L_ELU: O[j] = i > DU0
            ? (F[j] = DU1, i)
            : (F[j] = alpha * EXP(i)) - alpha;     break;
        case L_DROPOUT: O[j] = F[j] > alpha                /// * 1|0
            ? (F[j]=DU1, i)
            : (F[j]=DU0);                          break;
        }
    }
}
///
/// k_softmax_small — one block per sample, C ≤ 256
///
__KERN__ void k_softmax_small(DU *I, DU *O, int C) {
    __shared__ DU _smem[32];              ///< shared: partial max then partial sum
    __shared__ DU _max, _sum;             ///< block-wide max, sum

    const int CX = T4_DIM_SQ / 32;
    const int c  = threadIdx.x;           ///< channel index
    const int n  = blockIdx.x;            ///< sample index
    
    DU *s = &I[(long)n * C];              ///< slice for input n = blockIdx.x
    DU *d = &O[(long)n * C];              ///< slice for output n = blockIdx.x

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
        sm   = EXP(s[c] - _max);         /// * numerical stability: x - max
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

__KERN__ void k_softmax(DU *I, DU *O, int C) {
    __shared__ DU _smem[32];             ///< 8 warp results, padded to 32
    __shared__ DU _max, _sum;

    const int CX   = T4_DIM_SQ / 32;     ///< number of warps = 8
    const int c    = threadIdx.x;
    const int n    = blockIdx.x;         ///< sample index  ← fix
    const int step = blockDim.x;         ///< stride = T4_DIM_SQ = 256
    const int lane = c % 32;
    const int warp = c / 32;

    DU *s = &I[(long)n * C], *d = &O[(long)n * C];

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
        d[j] = EXP(s[j] - _max);
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

__KERN__ void k_batchnorm(
    DU *I, DU *O,  DU *X,                  ///< input, filter, output tensors
    DU *avg, DU *rvar,                     ///< mean, 1.0/(stdvar + e)
    DU *w, DU *b,                          ///< gamma, beta
    U64 HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const U64 j  = (U64)blockIdx.x * blockDim.x + threadIdx.x;  ///< element index
    const U32 c  = blockIdx.y, n = blockIdx.z, C = gridDim.y;   ///< channel deep, batch id
    const U64 k  = (HW * n + j) * C + c;                        ///< output tensor index

    if (j < HW) {
        O[k] = (X[k] = (I[k] - avg[c]) * rvar[c]) * w[c] + b[c];
    }
}
//
//< Neaural network forward propagation
// * input can be a Tensor or a Dataset
//
__HOST__ Model&
Model::forward(Tensor &input) {
    Tensor &n0 = (*this)[0];              ///< reference model input layer
    if (*_trace) input.show();            /// * preview input data

    if (input.numel != n0.numel) {
        ERROR("nn#forward dataset wrong shape[%d,%d,%d,%d] != model input[[%d,%d,%d,%d]\n",
              input.N(), input.H(), input.W(), input.C(),
              n0.N(), n0.H(), n0.W(), n0.C());
        return *this;
    }
    n0 = input;                 /// * copy dataset batch into the first layer [0,1)
    ///
    /// cascade execution layer by layer forward
    /// TODO: model execution becomes a superscalar pipeline
    ///
    auto info = [](DU t, int i, Tensor &in, Tensor &out) {
        INFO("\n%6.2f:%3d> %s [%2d,%2d,%2d,%2d] Σ/n=%6.2f p=%6.3f => out[%2d,%2d,%2d,%2d]",
            t, i, nname(in.grad_fn), in.N(), in.H(), in.W(), in.C(),
            in.sum() / in.N() / in.C(), in.xparm,
            out.N(), out.H(), out.W(), out.C());
    };
    NLOG("\nModel::forward starts trace=%d {", *_trace);
    DU t0 = System::clock(), t1 = t0, tt;           ///< performance measurement
    for (int i = 0; i < numel - 1; i++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (*_trace) {
            info((tt=System::clock()) - t1, i, in, out);
            t1 = tt;
        }
        _fstep(in, out);

        if (*_trace && _check_nan(out)) {
            ERROR("nn#forward Nan %s\n", nname(in.grad_fn));
            in.show();
            out.show();
            this->err = 1;
            break;
        }
        if (*_trace > 1) out.show();
    }
    ///
    /// collect onehot vector and hit count
    ///
    if (input.is_dataset()) {
        onehot((Dataset&)input);                   /// * update/cache onehot vector
        _hit = hit(true);                          /// * and _hit count
    }
    NLOG("\n} Model::forward %5.2f ms\n", System::clock() - t0);
    return *this;
}
/// ========================================================================
/// private methods
///
__HOST__ void
Model::_fstep(Tensor &in, Tensor &out) {
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                       ///< layer function
    switch(fn) {
    case L_CONV:    _fconv(in, out);         break; ///< convolution
    case L_LINEAR:  _flinear(in, out);       break; ///< out = W @ in + B
    case L_FLATTEN: out = in;                break; ///< straight copy
    case L_RELU:
    case L_TANH:
    case L_SIGMOID:
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:     _factivate(in, out, fn); break;
    case L_DROPOUT: {                               ///< dropout mask
        Tensor &t = *in.grad[4];
        System::rand(t.data, t.numel, UNIFORM);     /// * randomize w, shift pct
        _factivate(in, out, fn);
    } break;
    case L_SOFTMAX: _fsoftmax(in, out);      break; /// * feed to CrossEtropy
    case L_LOGSMAX: _flogsoftmax(in, out);   break; /// * feed to NLL
    case L_AVGPOOL:
    case L_MAXPOOL:
    case L_MINPOOL: _fpool(in, out, fn);     break;
    case L_BATCHNM: _fbatchnorm(in, out);    break;
    case L_USAMPLE: _fupsample(in, out);     break;
    default: ERROR("nn#fstep layer=%d not supported\n", fn);
    }
}

#define TSZ(r)    (T4_DIM_SZ - 2*(r))      /** 16,14,12,10 for r=0,1,2,3.. conv */
#define TILE(v,t) ((v) + (t) - 1)/(t)      /** dim of tile  */
#define CONV(r)                                                                 \
    k_conv2d<TSZ(r),(r)>                                                        \
    <<<dim3(TILE(W,TSZ(r)),TILE(H,TSZ(r)),C0*N), dim3(T4_DIM_SZ,T4_DIM_SZ,1)>>> \
    (in.data, f.data, b.data, out.data, H, W, C1, C0)

__HOST__ int
Model::_fconv(Tensor &in, Tensor &out) {
    Tensor &f = *in.grad[0], &b = *in.grad[1];            ///< filter (1x1, 3x3, 5x5, 7x7), bias tensor
    NN_DB(" f[%d,%d,%d,%d], b[%ld]", f.N(), f.H(), f.W(), f.C(), b.numel);

    const U32 N = out.N(), H = out.H(), W = out.W();      ///< outpt dimensions
    const U32 C0 = out.C(), C1 = in.C();                  ///< output, input channel deep

    U32 r  = (f.H() - 1) >> 1;
    switch(r) {
    case 0: CONV(0); break;                               ///< 1x1 (52 reg)
    case 1: CONV(1); break;                               ///< 3x3 (56 reg)
    case 2: CONV(2); break;                               ///< 5x5 (64 reg)
    case 3: CONV(3); break;                               ///< 7x7 (72 reg)
    default: ERROR("nn#fconv kernel_size=%d not supported\n", f.H()); return -1;
    }
    GPU_CHK();
    
    return 0;
}

__HOST__ int
Model::_flinear(Tensor &in, Tensor &out) {
    const U32 N  = out.N();                           ///< batch size (N1 == N0)
    const U32 E1 = in.HWC(), E0 = out.HWC();          ///< dense layer dims
    auto qa_calc = [&in, &out, N, E0, E1](DU *w, DU *b) {
        for (U32 n = 0; n < N; n++) {                 /// * walk through batch
            DU *x  = in.slice(n), *y = out.slice(n);  /// * sample by sample
            for (U32 e0 = 0; e0 < E0; e0++) {         /// * output features
                DU sum = b[e0];                       /// * init with bias
                printf("y[%d] = %g ", e0, sum);
                for (U32 e1 = 0; e1 < E1; e1++) {     /// * input features
                    sum += x[e1] * w[e0 * E1 + e1];   /// * Y = X @ W^T + B
                    printf(" +%g*%g=%g ", x[e1], w[e0*E1+e1], sum);
                }
                y[e0] = sum;
            }
        }
    };
    Tensor &w = *in.grad[0], &b = *in.grad[1];        ///< weight, bias tensors

    NN_DB(" = in[%d,%d,%d,%d] @ w[1,%d,%d,1]^T + b[%ld])",
          in.N(), in.H(), in.W(), in.C(), E0, E1, b.numel);
    
    if (*_trace > 1) {
        _dump_w("w", w, w.numel < T4_DIM_SQ);
        _dump_b("b", b);
    }

    if (0 && w.numel < T4_DIM_SQ) {                        /// * threshold control
        NN_DB("* in = "); in.show(true);
        qa_calc(w.data, b.data);                      /// * serial code
        NN_DB(" => out"); out.show(true);
    }
    else {
        // O[N,E0] = I[N,E1] @ W[E0,E1]^T + B[E0]
        // In your Tensor layout: A=in(H=N,W=E1,C=1), B=w(H=E0,W=E1,C=1)
        // tB=true transposes W from [E0,E1] to [E1,E0] for the multiply
        Tensor::linear(in, w, out, N, E0, E1,         /// * Y[N,E0] = X[N,E1] @ W[E0,E1]^T
                       DU1, DU0, false, true);
        FORK3(k_bias, N, E0, 1, b.data, out.data);    /// * Y[N,E0] += B[E0]
    }    
    return 0;
}

__HOST__ int
Model::_factivate(Tensor &in, Tensor &out, t4_layer fn) {
    DU alpha = in.xparm;
    FORK(k_activate, in.numel, 
         fn, in.data, in.grad[4]->data, out.data, alpha);
    return 0;
}

__HOST__ int
Model::_fpool(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W  = out.W(), H = out.H();                ///< output dimensions
    const U32 C  = out.C(), N = out.N();
    const int ks0= in.stride[0], ks1=in.stride[1];      ///< kernel size

    NN_DB(" %dx%d", ks0, ks1);
    
    switch(ks0) {                                       /// pooling kernel size
    case 2: FORK4(k_pool<2>, fn, in.data, out.data, H, W); break;
    case 3: FORK4(k_pool<3>, fn, in.data, out.data, H, W); break;
    default:
        ERROR("nn#fpool kernel_size=%d not supported\n", ks0);
        return -1;
    }
    return 0;
}

__HOST__ int
Model::_fsoftmax(Tensor &in, Tensor &out) {
    const U32 N = in.N();                       ///< batch size
    const U32 C = (U32)in.HWC();                ///< classes per sample = H*W*C
    
    if (C <= T4_DIM_SQ) {                       /// * one block per sample, all C classes fit in one block
        FORK2(k_softmax_small, N, C, in.data, out.data);
    }
    else {
        FORK2(k_softmax, N, C, in.data, out.data);
    }
    return 0;
}

__HOST__ int
Model::_flogsoftmax(Tensor &in, Tensor &out) {  /// * TODO: DCP
    Tensor &t = *in.grad[4];                    ///< temp tensor [1,H,W,C];
    DU     *d = t.data;                         ///< cache tensor data
    out = in;                                   /// * copy in data to out
    out.map(EXP);
    for (U32 n = 0; n < out.N(); n++) {         /// * loop throught mini-batch
        t.data = out.slice(n);
        DU sum    = t.sum();
        DU logsum = LOG(MAX(sum, DU_EPS));      ///< clamped
        t -= logsum;                            ///< xi - log(sum(exp(xi)))
    }
    t.data = d;                                 /// * restore tensor data pointer
    return 0;
}
///
///> batch norm, (batch mean=0.0, and variance=1.0)
///
__HOST__ int
Model::_fbatchnorm(Tensor &in, Tensor &out) {
    const U32 N   = out.N(), C = out.C(), H = out.H(), W = out.W(); ///< C0==C1
    const U64 HW  = (U64)H * W;                        ///< size of a page
    const U64 NHW = HW * N;                            ///< size of entire batch

    DU *w   = &in.grad[0]->data[0];                    ///< weight/gamma
    DU *b   = &in.grad[0]->data[C];                    ///< bias/beta
    DU *xht = in.grad[4]->data;                        ///< x_hat
    DU *avg = &in.mtum[4]->data[0];                    ///< mean
    DU *var = &in.mtum[4]->data[C];                    ///< 1.0/(sqrt(var)+e)

    for (U32 c=0; c < C; c++) avg[c] = var[c] = DU0;   /// * zero out
    FORK4(k_batchsum, in.data, avg, HW);               /// * capture sum

    for (U32 c=0; c < C; c++) avg[c] *= DU1 / NHW;     /// * calc mean per channel
    FORK4(k_batchnvar, in.data, avg, var, HW);         /// * capture n*variance

    const DU m = in.xparm;                             ///< ETA momentum, TODO:
    for (U32 c=0; c < C; c++) {
        var[c] = DU1 / (SQRT(var[c] / NHW) + DU_EPS);  ///< gvar = gamma/(stdvar + e)
    }
    FORK4(k_batchnorm, in.data, out.data, xht, avg, var, w, b, HW); /// * O = x_hat*gamma + beta
    return 0;
}
///
///> upsampling =~ reverse pooling (calls backprop k_dpool)
///
template<int KS>                                        /// forward declare (in backprop.cu)
__KERN__ void k_dpool(t4_layer op, DU *I, DU *O, U32 H, U32 W);
__HOST__ int
Model::_fupsample(Tensor &in, Tensor &out) {
    const U32 W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const U32 C  = in.C(), N = in.N();
    const int me = in.iparm;                            ///< upsample method, TODO
    const int ks = in.stride[0];                        ///< upsampling size

    switch(ks) {
    case 2: FORK4(k_dpool<2>, L_USAMPLE, out.data, in.data, H, W); break;
    case 3: FORK4(k_dpool<3>, L_USAMPLE, out.data, in.data, H, W); break;
    default:
        ERROR("nn#fupsample size=%d not supported\n", ks);
        return -1;
    }
    return 0;
}

} // namespace t4::nn

#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
