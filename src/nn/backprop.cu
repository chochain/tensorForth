/** -*- c++ -*-
 * @file
 * @brief Model class - backward propagation implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "nmath.h"

#if (T4_DO_OBJ && T4_DO_NN)

namespace t4::nn {
///
/// backprop: Neural Network back propegation
/// Note: cascade execution layer by layer backward
///
__HOST__ Model&
Model::broadcast(Tensor &tgt) {
    Tensor &out = (*this)[-1];                   ///< model output
    U64    HWC  = out.HWC();                     ///< sample size
    U32    N    = out.N();
    if (!_hot) _hot = &T4(N, 1, HWC, 1);         ///< allocate onehot vector if needed
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
    if (_bprep(tgt)) return *this;                        /// * dLoss pass-thru
    
    NLOG("\nModel::backprop starts trace=%d train=%d {", *_trace, train);
    DU  t0 = System::clock(), t1 = t0, tt;                ///< performance measurement
    for (int i = numel - 2, j = 0; i >= 0; i--, j++) {    ///< feed backward, skip last (output) layer
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (*_trace) {
            trace((tt=System::clock()) - t1, i, in, out);
            t1 = tt;
        }
        _bstep(in, out, j==0);

        if (*_trace && _check_nan(in)) {
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
Model::_bprep(Tensor &tgt) {                     ///> pre-calc dLoss pass-thru
    Tensor &out = (*this)[-1];                   ///< output layer, used as dLoss
    if (out.numel != tgt.numel) {                /// * check dimensions of target vector
        ERROR("Model#bprep: Onehot wrong shape[%d,%d,%d,%d] != [%d,%d,%d,%d], numel=%ld,%ld ",
              tgt.N(), tgt.H(), tgt.W(), tgt.C(),
              out.N(), out.H(), out.W(), out.C(), tgt.numel, out.numel);
        return 1;
    }
    
    NLOG("Model::bprep input(onehot) numel=%ld OK {", tgt.numel);
    
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
    case L_LINEAR:                               /// * linear  + MSE
    case L_SIGMOID:                              /// * sigmoid + BCE
    case L_SOFTMAX:                              /// * softmax + CE
    case L_LOGSMAX:                              /// * log-softmax + NLL
        out -= tgt;
        out *=  DU1 / tgt.N();                   /// * normalize match forward 1/N
        break;          
    default: out = tgt;  break;                  /// * pass thru pre-calc dLoss
    }
    if (*_trace) out.show();                     /// * display loss if trace on
    
    NLOG("}\n");

    return 0;
}

__HOST__ void
Model::_bstep(Tensor &in, Tensor &out, bool last_layer) {
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                       ///< layer function
    switch(fn) {
    case L_CONV:    _bconv(in, out);         break; /// * convolution
    case L_LINEAR:
        if (last_layer) in = out;                   /// * linear + MSE
        else            _blinear(in, out);   break; /// * out = w @ in + b
    case L_FLATTEN: in = out;                break; /// * pass dY to X
    case L_RELU:
    case L_TANH:                                    /// * in = (1 - t^2)*out
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:
    case L_DROPOUT: _bactivate(in, out);     break; /// * in = msk * out
    case L_SIGMOID:                                 /// * sigmoid + BCE
    case L_SOFTMAX:                                 /// * softmax + CrossEntropy (pass thru)
    case L_LOGSMAX: in = out;                break; /// * log-softmax + NLL (pass thru)
    case L_MAXPOOL:
    case L_AVGPOOL:
    case L_MINPOOL: _bpool(in, out, fn);     break;
    case L_BATCHNM: _bbatchnorm(in, out);    break;
    case L_USAMPLE: _bupsample(in, out, fn); break;
    case L_DCONV:   _fconv(in, out);         break;
    default: ERROR("nn#bstep layer=%d not supported\n", fn);
    }
}

#define TILE(v,t) (((v) + (t) - 1)/(t))            /** number of tiles */
#define DCONV(ks,s,p) do {                                              \
        constexpr int TS = (T4_DIM_SZ - (ks) + (s)) / (s);              \
        dim3 blk(T4_DIM_SZ, T4_DIM_SZ, 1);                              \
        dim3 grd(TILE(W0, TS), TILE(H0, TS), C0*C1*N);                  \
        k_dconv2d<TS, (ks), (s), (p)><<<grd,blk>>>(                     \
            in.data, out.data, dx.data, f.data, df.data, db.data,       \
            H1, W1, H0, W0, C0, C1, train);                             \
    } while(0)

__HOST__ int
Model::_bconv(Tensor &in, Tensor &out) {
    Tensor &f  = *in.grad[0], &df = *in.grad[2];
    Tensor &b  = *in.grad[1], &db = *in.grad[3];
    Tensor &dx = *in.grad[4];

    NN_DB(" f[%d,%d], b[%ld]", f.H(), f.W(), b.numel);

    const int N  = in.N(),  H1 = in.H(),  W1 = in.W();     ///< input dims
    const int H0 = out.H(), W0 = out.W();                  ///< output dims
    const int C1 = in.C(),  C0 = out.C();

    if (*_trace > 1) {
        _dump_b("before b",   b); _dump_f("before f",  f);
        _dump_b("before db", db); _dump_f("before df", df);
    }

    cudaMemset(dx.data, 0, dx.numel * sizeof(DU));         ///< pre-zero dX

    const U32 K = f.H();
    const U32 S = in.stride[0];                            ///< stride
    const U32 P = in.stride[2];                            ///< padding

    switch ((K<<8) | (S<<4) | P) {
    case 0x110: DCONV(1, 1, 0);  break;
    case 0x311: DCONV(3, 1, 1);  break;   /// * S=1, P=1 => same-size
    case 0x421: DCONV(4, 2, 1);  break;   /// * S=2, P=1 => same-size (ConvTranspose2D)
    case 0x512: DCONV(5, 1, 2);  break;   /// * S=1, P=2 => same size
    default:
        ERROR("nn#bconv kernel_size=%d stride=%d padding=%d not supported\n", K, S, P);
        return -1;
    }
    GPU_CHK();
    in = dx;                                                ///< x = dX (overwrite)

    if (*_trace > 1) {
        _dump_b("after db", db); _dump_f("after df", df);
    }
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
    if (train && *_trace > 1) {
        _dump_b("before db", db);
        _dump_w("before dw", dw, dw.numel < T4_DIM_SQ);
    }
    
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
    if (0 && w.numel < T4_DIM_SQ) {                   /// * threshold control
//        NN_DB("* out = "); out.show(true);
        qa_calc();                                    /// * serial mode (validation)
//        NN_DB(" => in"); in.show(true);
    }
    else {
        if (train) {
            FORK3(k_dlinear_db, N, E0, 1, out.data, db.data);    /// * dB += sum(dY)
            Tensor::linear(                           /// * dW[E0,E1] += dY[N,E0]^T @ X[N,E1]
                out, in, dw, E0, E1, N, DU1, DU1, true, false);
        }
        in.zeros();                                   /// * dX = 0
        Tensor::linear(                               /// * dX[N,E1] = dY[N,E0] @ W[E0,E1]
            out, w, in, N, E1, E0, DU1, DU0, false, false);
    }
    if (train && *_trace > 1) {
        _dump_b("after db", db);
        _dump_w("after dw", dw, dw.numel < T4_DIM_SQ);
    }
    return 0;
}

__HOST__ int
Model::_bactivate(Tensor &in, Tensor &out) {
    if (train && *_trace > 1) {
        _dump_f("msk", *in.grad[4]);
    }
    Tensor::ten_op(MUL, out, *in.grad[4], in);        /// * in = msk * out
    return 0;
}

__HOST__ int
Model::_bpool(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W = out.W(), H = out.H();               ///< output dimensions
    const U32 C = out.C(), N = out.N();
    const int K = in.stride[0];                      ///< kernel size (square)
    
    NN_DB(" %dx%d", K, K);
    
    switch(K) {
    case 2: FORK4(k_dpool<2>, 0, fn, in.data, out.data, H, W); break;
    case 3: FORK4(k_dpool<3>, 0, fn, in.data, out.data, H, W); break;
    default:
        ERROR("nn#bpool kernel_size=%d not supported\n", K);
        return -1;
    }
    return 0;
}
///
///> upsampling =~ reverse pooling (calls forward k_pool)
///
__HOST__ int
Model::_bupsample(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W  = in.W(), H = in.H();                ///< input dimensions (reversed pool)
    const U32 C  = in.C(), N = in.N();
    const int me = in.iparm;                          ///< upsample method, TODO
    const int K  = in.stride[0];                      ///< kernel size (square?)

    switch(K) {                                       /// by kernel size
    case 2: FORK4(k_pool<2>, 0, fn, out.data, in.data, H, W); break;
    case 3: FORK4(k_pool<3>, 0, fn, out.data, in.data, H, W); break;
    default:
        ERROR("nn#bupsample size=%d not supported\n", K);
        return -1;
    }
    return 0;
}
///
///> batchnorm
///  @brief:
///    see https://kevinzakka.github.io/2016/09/14/batch_normalization/
///
/// Launch sequence (was 4 kernels + 2 CPU loops; now 3 kernels, 0 CPU sync):
///
///   1. k_batchnorm_1 - fused Σdout and Σ(dout·x̂) in one tensor pass
///   2. k_batchnorm_2 - per-channel epilogue: accumulate dW/dB, scale sums
///   3. k_dbatchnorm  - fused dX update in one tensor pass
///
__HOST__ int
Model::_bbatchnorm(Tensor &in, Tensor &out) {
    const int  N   = out.N(), H = out.H(), W = out.W(), C = out.C();  ///< in==out
    const long HW  = (long)H * W;
    const long NHW = HW * N;

    DU *w   = in.grad[0]->data;              ///< gamma (scale)  [C]
    DU *dw  = in.grad[2]->data;              ///< d_gamma        [C]
    DU *b   = in.grad[1]->data;              ///< beta           [C] (not used)
    DU *db  = in.grad[3]->data;              ///< d_beta         [C]
    DU *xht = in.grad[4]->data;              ///< x_hat          [NHWC]
    DU *s1  = &in.mtum[4]->data[0];          ///< sum_dout       [NC]  (reused as s1 after scale)
    DU *s2  = &in.mtum[4]->data[N * C];      ///< sum_dout_xhat  [NC]  (reused as s2 after scale)
    DU *var = &in.mtum[4]->data[N * C * 2];  ///< 1/sqrt(var+e)  [C]

    // zero the accumulators before reduction
    cudaMemset(s1, 0, N * C * 2 * sizeof(DU));

    /// 1. fused reduction ---
    {
        const int nwarps  = (T4_DIM_SQ + 31) >> 5;
        const int smem_sz = 2 * nwarps * sizeof(DU);
        FORK4(k_dbatchnorm_1, smem_sz, out.data, xht, s1, s2, HW);
    }
    /// 2. per-channel scale (no CPU sync needed) ---
    //   launched as <<<C, 1>>> so each channel is one thread in one block;
    //   gridDim.x == C is used inside the kernel as the channel stride for [N*C].
    {
        dim3 _b(C, 1, 1);
        k_dbatchnorm_2<<<1, _b>>>(
            w, dw, db, s1, s2, var, N, NHW, train);
        GPU_CHK();
    }
    /// 3. fused dX update ---
    FORK4(k_dbatchnorm, 0, in.data, out.data, xht, s1, s2, HW);

    return 0;
}

} // namespace t4::nn

#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
