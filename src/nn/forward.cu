/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <float.h>
#include "model.h"
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
#if (T4_DO_OBJ && T4_DO_NN)

namespace t4::nn {
//
//< Neaural network forward propagation
// * input can be a Tensor or a Dataset
//
__HOST__ Model&
Model::forward(Tensor &input) {
    Tensor &n0 = (*this)[0];              ///< reference model input layer
    if (*_trace) input.show();            /// * preview input data

    if (input.numel != n0.numel) {
        ERROR("nn#forward dataset wrong shape[%d,%d,%d,%d] != model input[%d,%d,%d,%d]\n",
              input.N(), input.H(), input.W(), input.C(),
              n0.N(), n0.H(), n0.W(), n0.C());
        return *this;
    }
    n0 = input;           /// * copy tensor or dataset batch into the first layer [0,1)
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
            ERROR("nn#forward Nan in %s\n", nname(in.grad_fn));
            INFO("in=");  in.show(true);
            INFO("out="); out.show(true);
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
    case L_DCONV:   _bconv(in, out);         break;
    default: ERROR("nn#fstep layer=%d not supported\n", fn);
    }
}

#define TILE(v,t) (((v) + (t) - 1)/(t))     /** number of tiles */
#define CONV(ks,s,p) do {                                       \
    constexpr int TS = (T4_DIM_SZ - (ks) + (s)) / (s);          \
    dim3 blk(T4_DIM_SZ, T4_DIM_SZ, 1);                          \
    dim3 grd(TILE(W0, TS), TILE(H0, TS), C0 * C1 * N);          \
    k_conv2d<TS, (ks), (s), (p)><<<grd,blk>>>(                  \
        in.data, out.data, f.data, b.data,                      \
        H1, W1, H0, W0, C1, C0);                                \
    } while(0)

__HOST__ int
Model::_fconv(Tensor &in, Tensor &out) {
    Tensor &f = *in.grad[0], &b = *in.grad[1];    ///< filter, bias tensors
    NN_DB(" f[%d,%d], b[%ld]", f.H(), f.W(), b.numel);

    const U32 N  = out.N(), H0 = out.H(), W0 = out.W(); ///< output dimensions
    const U32 H1 = in.H(),  W1 = in.W();                ///< input dimensions
    const U32 C0 = out.C(), C1 = in.C();                ///< output/input channels

    const U32 KS = f.H();                               ///< kernel side length
    const U32 S  = in.stride[0];                        ///< stride (stored in Tensor.stride[0])
    const U32 P  = in.stride[2];                        ///< padding (stored in Tensor.stride[2])
    ///
    /// dispatch on (kernel size, stride, padding) triples.
    ///
    switch ((KS << 8) | (S<<4) |  P) {
    case 0x110: CONV(1, 1, 0);   break;
    case 0x311: CONV(3, 1, 1);   break;   /// * S=1, P=1 => same-size
    case 0x421: CONV(4, 2, 1);   break;   /// * S=2, P=1 => same-size (ConvTranspose2D)
    case 0x512: CONV(5, 1, 2);   break;   /// * S=1, P=2 => same size
    default: {
        ERROR("nn#fconv kernel_size=%d stride=%d padding=%d not supported\n", KS, S, P);
        return -1;
    }
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
    Tensor &w  = *in.grad[0], &b  = *in.grad[1];        ///< weight, bias tensors

    NN_DB(" = in[%d,%d] @ w[%d,%d]^T + b[%ld])", in.H(), in.W(), E0, E1, b.numel);
    
    if (*_trace > 1) {
        _dump_w("w", w, w.numel < T4_DIM_SQ);
        _dump_b("b", b);
    }

    if (0 && w.numel < T4_DIM_SQ) {                        /// * threshold control
//        NN_DB("* in = "); in.show(true);
        qa_calc(w.data, b.data);                      /// * serial code
//        NN_DB(" => out"); out.show(true);
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
         fn, in.data, out.data, in.grad[4]->data, alpha);
    if (train && *_trace > 1) {
        _dump_f("msk", *in.grad[4]);
    }
    return 0;
}

__HOST__ int
Model::_fpool(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W = out.W(), H = out.H();                ///< output dimensions
    const U32 C = out.C(), N = out.N();
    const int K = in.stride[0];                        ///< kernel (TODO: rectangle)

    NN_DB(" %dx%d", K, K);
    
    switch(K) {                                        /// pooling kernel size
    case 2: FORK4(k_pool<2>, 0, fn, in.data, out.data, H, W); break;
    case 3: FORK4(k_pool<3>, 0, fn, in.data, out.data, H, W); break;
    default:
        ERROR("nn#fpool kernel_size=%d not supported\n", K);
        return -1;
    }
    return 0;
}

__HOST__ int
Model::_fsoftmax(Tensor &in, Tensor &out) {
    const U32 N = in.N();                       ///< batch size
    const U32 C = (U32)in.HWC();                ///< classes per sample = H*W*C
    /// * Note: grad[4] mask is not used
    
    if (C <= T4_DIM_SQ) {                       /// * one block per sample, all C classes fit in one block
        FORK2(k_softmax_small, N, C, in.data, out.data);
    }
    else {
        FORK2(k_softmax, N, C, in.data, out.data);
    }
    return 0;
}

__HOST__ int
Model::_flogsoftmax(Tensor &in, Tensor &out) {  
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
    const cudaStream_t st = 0;
    const U32 N   = out.N(), C = out.C(), H = out.H(), W = out.W();
    const U64 HW  = (U64)H * W;
    const U64 NHW = HW * N;

    DU *w   = in.grad[0]->data;              ///< gamma  [C]
    DU *b   = in.grad[1]->data;              ///< beta   [C]
    DU *xht = in.grad[4]->data;              ///< x_hat  [in.NHWC]
    DU *avg = &in.mtum[4]->data[0];          ///< avg    [C] - tmp, also s1/s2 in backprop
    DU *var = &in.mtum[4]->data[N * C * 2];  ///< var    [C] - read by backprop as rvar

    cudaMemsetAsync(avg, 0, C * 2 * sizeof(DU), st);  ///< zeros avg, var in one call

    /// 1. accumulate Σx and Σx² per channel
    {
        const int  _b = (int)MAX(32LL, MIN(HW, (U64)1024));
        const dim3 _g(C, N, 1);
        const int  smem_sz  = 2 * _b * sizeof(DU);
        k_batchnorm_stat<<<_g, _b, smem_sz, st>>>(in.data, avg, var, HW);
        GPU_CHK();
    }
    /// 2. finalise mean and rvar — no CPU round-trip
    {
        const int _b = MIN((int)C, 1024);
        const int _g = ((int)C + _b - 1) / _b;
        k_batchnorm_calc<<<_g, _b, 0, st>>>(avg, var, (long)NHW);
        GPU_CHK();
    }
    /// 3. apply normalisation
    FORK4(k_batchnorm, 0, in.data, out.data, xht, avg, var, w, b, HW);  ///< TODO: pass stream
    return 0;
}
///
///> upsampling =~ reverse pooling (calls backprop k_dpool)
///
__HOST__ int
Model::_fupsample(Tensor &in, Tensor &out) {
    const U32 W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const U32 C  = in.C(), N = in.N();
    const int me = in.iparm;                            ///< upsample method, TODO
    const int K  = in.stride[0];                        ///< upsampling size

    switch(K) {
    case 2: FORK4(k_dpool<2>, 0, L_USAMPLE, out.data, in.data, H, W); break;
    case 3: FORK4(k_dpool<3>, 0, L_USAMPLE, out.data, in.data, H, W); break;
    default:
        ERROR("nn#fupsample size=%d not supported\n", K);
        return -1;
    }
    return 0;
}

} // namespace t4::nn

#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
