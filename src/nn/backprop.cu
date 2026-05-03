/** -*- c++ -*-
 * @file
 * @brief Model class - backward propagation implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)

namespace t4::nn {
///
/// convolution filter derivatives
/// TODO: stride, dilation, [C1]NCHW filter
///
template<int TS, int R>    ///> tile size, kernel radius
__KERN__ void k_dconv2d(
    DU *I, DU *DX, DU *F, DU *DF, DU *DB, DU *O,     ///< input I[HxW], F,DF[KSxKS], output O[HxW]
    int H, int W, int C0, bool train                 ///< H1==H0, W1==W0, output Channels
    ) {
    constexpr int KS  = 2 * R + 1;
    constexpr int SSZ = TS + KS - 1;
    __shared__ DU _I[SSZ][SSZ];                      ///< input cache tile
    __shared__ DU _O[SSZ][SSZ];                      ///< output cache tile
    __shared__ DU _df[KS][KS];

    const int  tx = threadIdx.x, j1 = tx + blockIdx.x * TS;
    const int  ty = threadIdx.y, i1 = ty + blockIdx.y * TS;
    const int  c1 = blockIdx.z,  C1 = gridDim.z;     ///< channel deep
    const int  load_id = ty * TS + tx;
    ///
    /// process z1, i.e. [TS, TS, C1] cells per kernel call
    ///
    const int zi = blockIdx.y * TS - R, zj = blockIdx.x * TS - R;
    for (int t = load_id; t < SSZ*SSZ; t += TS*TS) { /// * cached X (input) tile
        int si = t / SSZ, sj = t % SSZ;
        int gi = zi + si, gj = zj + sj;
        _I[si][sj] = (gi>=0 && gi<H && gj>=0 && gj<W)
            ? I[((long)W * gi + gj) * C1 + c1] : DU0;
    }
    ///
    /// for each output channel
    ///
    for (int c0 = 0; c0 < C0; c0++) {                ///< each dY channel
        for (int t=load_id; t < SSZ*SSZ; t += TS*TS) {
            int si = t / SSZ, sj = t % SSZ;
            int gi = zi + si, gj = zj + sj;
            _O[si][sj] = (gi >=0 && gi < H && gj >=0 && gj < W)
                ? O[((long)W * gi + gj) * C0 + c0] : DU0;               /// * cache input data
        }
        if (tx < KS && ty < KS) _df[ty][tx] = DU0;   /// * _df[KS][KS]
        __syncthreads();
        
        /// dB[c0] += dY[c0]
        DU db = (train && tx < TS && ty < TS) ? _O[ty+R][tx+R] : DU0;
        WARP_REDUCE(db);
        if ((ty * blockDim.x + tx) % 32 == 0) {
            atomicAdd(&DB[c0], db);
        }
        
        const U64 zf = (U64)C0 * KS * KS * c1 + c0;      ///< filter index F[C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                        /// * within tile [14x14]
             DU sum = DU0;                               ///< dX sum, DF[c1,0,0,c0] (TSxTS threads)
             for (int y=0, y1=KS-1; y < KS; y++, y1--) {
                 for (int x=0, x1=KS-1; x < KS; x++, x1--) {
                    int fi = zf + (y1 * KS + x1) * C0;   ///< rot180 index
                    DU  dy = _O[ty+y][tx+x];
                    sum += F[fi] * dy;                   /// * dX = F @ dY
                    if (train) {                         /// * dF = dY @ X
                        atomicAdd(&_df[y][x], dy * _I[ty+y][tx+x]); 
                    }
                }
            }
            if (i1 < H && j1 < W) {                       /// * collect dX (per C1)
                DX[((long)W * i1 + j1) * C1 + c1] += sum; ///< final dX
            }
        }
        __syncthreads();

        if (tx < KS && ty < KS) {
            atomicAdd(&DF[zf + (ty * KS + tx) * C0], _df[ty][tx]); /// * dF += sum(_df)
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// k_dlinear_db — add O[N, E0] to each dB[E0]
// ---------------------------------------------------------------------------
__KERN__ void k_dlinear_db(DU *O, DU *DB, int N, int E0) {
    const int e0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n  = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e0 < E0 && n < N) atomicAdd(&DB[e0], O[n * E0 + e0]);
}

template<int KS>                                      /// kernel size
__KERN__ void k_dpool(
    t4_layer op,
    DU *I, DU *O,                                     ///< input, output buffers
    U32 H, U32 W                                      ///< output HW (C1==C0)
    ) {
    const U32 KSQ= KS * KS;
    const U64 HW = (U64)H * W;                        ///< HxW
    const U64 k0 = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    const U32 j0 = k0 % W;                            ///< output x dim
    const U32 c  = blockIdx.y, C = gridDim.y;         ///< channel deep
    const U64 ns = HW * C * blockIdx.z;               ///< batch slice idx
    const U64 z0 = (U64)C * k0 + ns + c;              ///< output array index
    const U64 z1 = (U64)C * KS * j0 + KSQ * ((k0 - j0) * C + ns) + c;

    if (k0 < HW && c < C) {
        const U64 RI = (U64)KS * C * (W - 1);         ///< input cell row increment
        DU *ix = &I[z1], *t = ix;                     /// *ix input tensor cell
        DU2 v  = (op != L_AVGPOOL) ? *ix : O[z0] / KSQ;
        for (U32 y = 0; y < KS; y++) {     /// * handle one kernel
            for (U32 x = 0; x < KS; x++) {
                DU dx = *ix;
                switch (op) {
                case L_AVGPOOL: *ix = v;             break;
                case L_MAXPOOL:
                    *ix = DU0;             /// * zero out all elements
                    if (dx > v) { v = dx; t = ix; }  break;
                case L_MINPOOL:
                    *ix = DU0;
                    if (dx < v) { v = dx; t = ix; }  break;
                case L_USAMPLE: *ix = O[z0];         break;
                }
                ix += C;                   /// * next cell
            }
            ix += RI;                      /// * next input row
        }
        if (op==L_MAXPOOL || op==L_MINPOOL) *t = O[z0];    /// * update arg cell
    }
}

__KERN__ void k_dbatchnorm_1(
    DU *I, DU *O, DU *X,                   ///< input, output, x_hat tensors
    DU *sum, DU *g_var,                    ///< sum(x_hat), gamma/(stdvar+e)
    U64 HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const U64 j  = (U64)threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const U32 c  = blockIdx.y, C = gridDim.y;                   ///< channel deep
    const U32 n  = blockIdx.z, nc = C * n + c;                  ///< batch_id, sum/var index
    const U64 ns = HW * C * n;                                  ///< batch slice index
    const U64 k  = (U64)C * j + ns + c;                         ///< output tensor index
    const DU  _N = DU1 / gridDim.z;                             ///< 1.0/HWN

    if (c < C && j < HW) {
        I[k] = (O[k] - sum[nc] * _N) * g_var[nc];               /// * dX = g_var * (dout - sum(dout) / N)
        O[k] *= X[k];                                           /// * dout * x_hat
    }
}
__KERN__ void k_dbatchnorm_2(
    DU *I, DU *X, DU *sum,                 ///< input, x_hat
    U64 HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const U64 j  = (U64)blockIdx.x * blockDim.x + threadIdx.x;  ///< element index
    const U32 c  = blockIdx.y, C = gridDim.y;                   ///< channel deep
    const U32 n  = blockIdx.z, nc = C * n + c;                  ///< batch_id, sum index
    const U64 ns = HW * C * n;                                  ///< batch slice index
    const U64 k  = (U64)C * j + ns + c;                         ///< output tensor index

    if (c < C && j < HW) I[k] -= X[k] * sum[nc];
}
///
/// backprop: Neural Network back propegation
/// Note: cascade execution layer by layer backward
///
__HOST__ Model&
Model::broadcast(Tensor &tgt) {
    Tensor &out = (*this)[-1];                   ///< model output
    U64    HWC  = out.HWC();                     ///< sample size
    U32    N    = out.N();
    if (!_hot) _hot = &T4(N, HWC);               ///< allocate onehot vector if needed
    for (U32 n = 0; n < N; n++) {                /// * loop through batch, TODO: Kernel
        DU  v = tgt.data[n];                     ///< target vector
        DU *h = _hot->slice(n);                  ///< take a sample
        for (U64 i=0; i<HWC; i++) h[i] = v;      /// * broadcast [N,1] => [N,HWC]
    }
    return *this;
}

__HOST__ Model&
Model::backprop() {
    if (_hot) return backprop(*_hot);            /// * use default one-hot vector

    ERROR("nn#backprop missing onehot vector?\n");
    return *this;
}

__HOST__ Model&
Model::backprop(Tensor &tgt) {
    auto trace = [](DU t, int i, Tensor &in, Tensor &out) {
        INFO("\n%6.2f:%3d> %s [%2d,%2d,%2d,%2d] p=%6.3f <= out'Σ/n=%6.2f [%2d,%2d,%2d,%2d]",
            t, i, nname(in.grad_fn),
            in.N(), in.H(), in.W(), in.C(), in.xparm,
            out.sum() / out.N() / out.C(),
            out.N(), out.H(), out.W(), out.C());
    };
    if (_bloss(tgt)) return *this;                        /// * pre-calculate dLoss
    
    NLOG("\nModel#backprop starts {");
    DU  t0 = System::clock(), t1 = t0, tt;                ///< performance measurement
    for (int i = numel - 2, j = 0; i >= 0; i--, j++) {    ///< feed backward, skip last (output) layer
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (*_trace) {
            trace((tt=System::clock()) - t1, i, in, out);
            t1 = tt;
        }
        _bstep(in, out);

        if (_check_nan(in)) {
            ERROR("nn#backprop Nan %s\n", nname(in.grad_fn));
            in.show();
            out.show();
            this->err = 1;
            break;
        }
        if (*_trace > 1) in.show();
    }
    NLOG("\n} Model::backprop %5.2f ms\n", System::clock() - t0);
    return *this;
}
/// ========================================================================
/// private methods
///
__HOST__ int
Model::_bloss(Tensor &tgt) {                     ///> pre-calc dLoss
    Tensor &out = (*this)[-1];                   ///< output layer, used as dLoss
    if (out.numel != tgt.numel) {                /// * check dimensions of target vector
        ERROR("Model#bloss: Onehot wrong shape[%d,%d,%d,%d] != [%d,%d,%d,%d], numel=%ld,%ld ",
              tgt.N(), tgt.H(), tgt.W(), tgt.C(),
              out.N(), out.H(), out.W(), out.C(), tgt.numel, out.numel);
        return 1;
    }
    
    NLOG("Model::bloss input(onehot) numel=%ld OK {", tgt.numel);
    
    t4_layer fn = (*this)[-2].grad_fn;           ///< final activation layer
    ///
    /// * NN typically utilize the following functions as final layer.
    ///   Their derivative, when associated with specific loss functions,
    ///   can be treated as pass-thru 
    ///
    ///   + sigmod + BCE (BinaryCrossEntropy) loss for binary categorization
    ///   + softmax + CE (CrossEntropy) loss for multi-class categorization
    ///   + log-softmax + NLL (Negative Log Likelihood) for multi-class
    ///
    switch (fn) {
    case L_SIGMOID:                              /// * sigmoid + BCE
    case L_SOFTMAX:                              /// * softmax + CE
    case L_LOGSMAX: out -= tgt;  break;          /// * log-softmax + NLL
    default:        out  = tgt;  break;          /// * pass thru pre-calc dLoss, i.g. MSE
    }
    if (*_trace) out.show();                     /// * display loss if trace on
    
    NLOG("}\n");

    return 0;
}

__HOST__ void
Model::_bstep(Tensor &in, Tensor &out) {
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                       ///< layer function
    switch(fn) {
    case L_CONV:    _bconv(in, out);         break; /// * convolution
    case L_LINEAR:  _blinear(in, out);       break; /// * out = w @ in + b
    case L_FLATTEN: in = out;                break; /// * pass dY to X
    case L_RELU:
    case L_TANH:                                    /// * in = (1 - t^2)*out
    case L_SIGMOID:                                 /// * in = s*(1 - s)*out
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:
    case L_DROPOUT: _bactivate(in, out);     break; /// * in = msk * out
    case L_SOFTMAX:                                 /// * softmax + CrossEntropy (pass thru)
    case L_LOGSMAX: in = out;                break; /// * log-softmax + NLL (pass thru)
    case L_MAXPOOL:
    case L_AVGPOOL:
    case L_MINPOOL: _bpool(in, out, fn);     break;
    case L_BATCHNM: _bbatchnorm(in, out);    break;
    case L_USAMPLE: _bupsample(in, out, fn); break;
    default: ERROR("nn#bstep layer=%d not supported\n", fn);
    }
}

#define TSZ(r)    (T4_DIM_SZ - 2*(r))      /** 16,14,12 for 1x1,3x3,5x5 conv */
#define TILE(v,t) ((v) + (t) - 1)/(t)      /** dim of tile  */
#define DCONV(r)                                                              \
    k_dconv2d<TSZ(r),(r)>                                                     \
    <<<dim3(TILE(W,TSZ(r)),TILE(H,TSZ(r)),C0), dim3(T4_DIM_SZ,T4_DIM_SZ,1)>>> \
    (d1, dx, f.data, df.data, db.data, d0, H, W, C0, train)

__HOST__ int
Model::_bconv(Tensor &in, Tensor &out) {
    Tensor &f = *in.grad[0], &df = *in.grad[2];      ///< filter tensors
    Tensor &b = *in.grad[1], &db = *in.grad[3];      ///< bias tensors
    Tensor &x = *in.grad[4];                         ///< dX storage

    NN_DB(" f[%d,%d,%d,%d], b[%ld]", f.N(), f.H(), f.W(), f.C(), b.numel);

    const int N = in.N(), H = in.H(), W = in.W();    ///< input dimensions
    const int C1 = in.C(), C0 = out.C();
    
    if (*_trace > 1) { _dump_b("b", b); _dump_f("f", f); }
    
    x.zeros();
    for (int n = 0; n < N; n++) {                    ///< accumulative over N samples TODO: multi-stream
        DU *d1 = in.slice(n), *dx = x.slice(n), *d0 = out.slice(n);
        const int r = (f.H() - 1) >> 1;              ///< kernel radius
        switch (r) {
        case 0: DCONV(0); break;
        case 1: DCONV(1); break;
        case 2: DCONV(2); break;
        case 3: DCONV(3); break;
        default: ERROR("nn#bconv kernel_size %d not supported\n", f.H()); return -1;
        }
        GPU_CHK();
    }
    Tensor::copy(x, in);                             /// * X = dX (overwrite)
    
    if (*_trace > 1) { _dump_b("db", db); _dump_f("df", df); }
    return 0;
}

__HOST__ int
Model::_blinear(Tensor &in, Tensor &out) {
    Tensor &w = *in.grad[0], &dw = *in.grad[2];       ///< weight tensors
    Tensor &b = *in.grad[1], &db = *in.grad[3];       ///< bias tensors
    const int  N  = in.N(),   C0 = w.H(), C1 = w.W(); ///< dimensions
    const long E1 = in.HWC(), E0 = out.HWC();         ///< input, output element counts (for validation)
    
    NN_DB("\n\tdw[%d,%d] += out'[%ld,1] @ in^t[1,%ld]", C0, C1, E0, E1);
    NN_DB("\n\tin[%ld,1] = w^t[%d,%d] @ out'[%ld,1]", E1, C1, C0, E0);
    
    auto qa_calc = [&]() {
        for (int n = 0; n < in.N(); n++) {            ///< acc over N samples
            DU *x = in.slice(n), *y = out.slice(n);
            if (train) {
                DU *dp = dw.data;
                for (int e0 = 0; e0 < E0; e0++) {     /// W[E0,E1]
                    DU dy = y[e0];
                    db[e0] += dy;                     /// * db += dY
                    for (int e1 =0; e1 < E1; e1++) {
                        *dp++ += dy * x[e1];          /// * dw += dY^t @ X
                    }
                }
            }
            DU *wd = w.data;
            for (int e1 = 0; e1 < E1; e1++) {         /// * dX = w @ dY
                DU sum = DU0;
                for (int e0 = 0; e0 < E0; e0++) {
                    sum += wd[E1 * e0 + e1] * y[e0];
                }
                x[e1] = sum;
            }
        }
    };
    if (w.numel < T4_DIM_SQ) {                        /// * threshold control
        NN_DB("* out = "); out.show(true);
        qa_calc();                                    /// * serial mode (validation)
        NN_DB(" => in"); in.show(true);
    }
    else {
        if (train) {
            db.zeros();                                          /// * pre-zero dB
            FORK3(k_dlinear_db, N, E0, 1, out.data, db.data);    /// * dB += sum(dY)
            Tensor::linear(out, in, dw, E0, E1, N,               /// * dW[E0,E1] = dY[N,E0]^T @ X[N,E1]
                           DU1, DU0, true, false);
        }
        in.zeros();                                              /// * dX = 0
        Tensor::linear(out, w, in, N, E1, E0,                    /// * dX[N,E1] = dY[N,E0] @ W[E0,E1]
                       DU1, DU0, false, false);
    }
    if (train && *_trace > 1) {
        _dump_b("db", db);
        _dump_w("dw", dw, true);
    }
    return 0;
}

__HOST__ int
Model::_bactivate(Tensor &in, Tensor &out) {
    Tensor::ten_op(MUL, out, *in.grad[4], in);     /// * in = msk * out
    return 0;
}

__HOST__ int
Model::_bpool(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W = out.W(), H = out.H();           ///< output dimensions
    const U32 C = out.C(), N = out.N();
    const int ks = in.stride[0];                  ///< kernel size (square)
    switch(ks) {
    case 2: FORK4(k_dpool<2>, fn, in.data, out.data, H, W); break;
    case 3: FORK4(k_dpool<3>, fn, in.data, out.data, H, W); break;
    default:
        ERROR("nn#bpool kernel_size=%d not supported\n", ks);
        return -1;
    }
    return 0;
}
///
///> upsampling =~ reverse pooling (calls forward k_pool)
///
template<int KS>                                        /// forward declare (in forward.cu)
__KERN__ void k_pool(t4_layer op, DU *I, DU *O, U32 H, U32 W);
__HOST__ int
Model::_bupsample(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const U32 C  = in.C(), N = in.N();
    const int me = in.iparm;                            ///< upsample method, TODO
    const int ks = in.stride[0];                        ///< kernel size (square?)

    switch(ks) {                                        /// by kernel size
    case 2: FORK4(k_pool<2>, fn, out.data, in.data, H, W); break;
    case 3: FORK4(k_pool<3>, fn, out.data, in.data, H, W); break;
    default:
        ERROR("nn#bupsample size=%d not supported\n", ks);
        return -1;
    }
    return 0;
}
///
///> batchnorm
///  @brief:
///    see https://kevinzakka.github.io/2016/09/14/batch_normalization/
///  @note
///    my own implmentation having dbeta and dgamma divided by HW
///    which is different from original document by does better
///    in preventing gradient explosion
///
__HOST__ int
Model::_bbatchnorm(Tensor &in, Tensor &out) {
    const U32 N = out.N(), H = out.H(), W = out.W(), C = out.C();   ///< C0==C1, N1=N0
    const U64 HW = (U64)W * H, NHW = HW * N;

    DU *w   = &in.grad[0]->data[0];                    ///< weight/gamma (scale)
    DU *dw  = &in.grad[2]->data[0];                    ///< d_gamma
    DU *db  = &in.grad[2]->data[C];                    ///< d_beta
    DU *xht = in.grad[4]->data;                        ///< x_hat
    DU *sum = &in.mtum[4]->data[0];                    ///< batch sum
    DU *var = &in.mtum[4]->data[C];                    ///< batch 1.0 / (var+e)^0.5

    FORK4(k_batchsum, out.data, sum, HW);              /// * capture out sum(dout)     
    
    for (U32 c=0; c < C; c++) {
        if (train) db[c] += (sum[c] /= NHW);           /// * collect dbeta = sum(dout) (/ HW?)
        var[c] *= w[c];                                /// * var <= gamma * ivar
    }
    FORK4(k_dbatchnorm_1,                              /// * dX = gamma*ivar*(dout - sum(dout)/N)
        in.data, out.data, xht, sum, var, HW);         /// * also, dout *= x_hat
    
    FORK4(k_batchsum, out.data, sum, HW);              /// * capture sum(dout * x_hat)

    for (U32 c=0; c < C; c++) {
        if (train) dw[c]  += (sum[c] /= NHW);          /// * collect dgamma = sum(dout * x_hat)( / HW?)
        sum[c] *= var[c] / N;                          /// * scale sum
    }
    FORK4(k_dbatchnorm_2, in.data, xht, sum, HW);      /// * dX -= gamma*ivar*x_hat*sum(dout * x_hat) / N
    
    return 0;
}

} // namespace t4::nn

#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
