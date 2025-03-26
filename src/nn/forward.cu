/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)
///
/// convolution filter
/// TODO: stride, dilation, [C1]NCHW filter
///
template<int TS, int KS>         ///> tile size, kernel size
__KERN__ void k_conv2d(
    DU *I, DU *F, DU *B, DU *O,  ///> input I[HxW], F[KxK] kernel, B[C] bias, output O[HxW]
    U32 H, U32 W, U32 C1         ///< (H0==H1, W0==W1), input channels
    ) {
    __shared__ DU _I[T4_DIM_SZ][T4_DIM_SZ];          ///< shared memory [16x16]

    const U32 tx = threadIdx.x, j0 = tx + blockIdx.x * TS;   ///< output coordinates
    const U32 ty = threadIdx.y, i0 = ty + blockIdx.y * TS;   /// * i0,j0=0:15
    const U32 c0 = blockIdx.z,  C0 = gridDim.z;      ///< channel deep
    const U64 z0 = ((U64)W * i0 + j0) * C0 + c0;     ///< output array index
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const U32 KSQ= KS * KS;
    const int i1 = i0 - (KS >> 1);                   ///< input coordinates
    const int j1 = j0 - (KS >> 1);                   /// * i1,j1=-1:14

    auto g = cg::this_thread_block();                ///< all threads of block
    for (U32 c1 = 0; c1 < C1; c1++) {                ///< each input channel
        const U64 z1 = ((U64)W * i1 + j1) * C1 + c1; ///< one channel at a time
        _I[ty][tx] =                                 /// * cache input data
            (i1 >= 0 && i1 < H && j1 >= 0 && j1 < W) /// * with zero padding
            ? I[z1] : DU0;                           /// * by channel
        g.sync();                                    /// * smem write barrier
        ///
        /// Y = sum(W * X)
        /// TODO: cache F
        ///
        const U64 zf = (U64)C0 * KSQ * c1 + c0;      ///< filter index [C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                    /// * each tile
            DU sum = DU0, *fx = &F[zf];              ///< sum and filter[0]
            #pragma unroll
            for (int y = 0; y < KS; y++) {           /// * process one KS * KS cell
                DU *ix = &_I[ty+y][tx];              ///< X tile
                for (int x = 0; x < KS; x++) {
                    sum += (*fx) * ix[x];            /// Y += W * X
                    fx  += C0;                       /// * next filter cell
                }
            }
            if (i0 < W && j0 < H) {
                if (c1==0) O[z0] = sum + B[c0];      /// * O[ijc] with bias
                else       O[z0] += sum;
            }
        }
        g.sync();
    }
}

__KERN__ void k_linear(
    DU *I, DU *O, DU *W, DU *B,
    U64 HWC1, U64 HWC0, U32 C1, U32 C0
    ) {
    const U32 c1 = blockIdx.x * blockDim.x + threadIdx.x;
    const U32 c0 = blockIdx.y * blockDim.y + threadIdx.y;
    const U32 n  = blockIdx.z;

    if (c0 < C0 && c1 < C1) {
        DU *y = &O[HWC0 * n + c0];
        if (c1 == 0) *y = B[c0];                      /// Y = WX + B
        atomicAdd_block(y, W[C1 * c0 + c1] * I[HWC1 * n + c1]);
    }
}

template<int KS>                                      /// kernel size
__KERN__ void k_pool(
    t4_layer op,                                      ///< pooling ops
    DU *I, DU *O,                                     ///< input, output buffers
    U32 H, U32 W                                      ///< output HW (C0==C1)
    ) {
    const U64 HW = (U64)H * W;                        ///< output dimension
    const U64 k0 = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    const U32 KSQ= KS * KS;
    const U32 j0 = k0 % W;                            ///< output x dim
    const U32 c  = blockIdx.y, C = gridDim.y;         ///< channel deep
    const U64 ns = HW * C * blockIdx.z;               ///< batch slice idx
    const U64 z0 = (U64)C * k0 + ns + c;              ///< output array index
    const U64 z1 = (U64)C * KS * j0 + ((k0 - j0) * C + ns) * KSQ + c;
    const U64 RI = ((U64)W - 1) * KS * C;             ///< input cell row increment
    const bool avg = (op != L_MAXPOOL && op != L_MINPOOL);

    if (k0 < HW && c < C) {
        DU *ix = &I[z1];
        DU2 v  = avg ? DU0 : *ix;
        #pragma unroll
        for (int y = 0; y < KS; y++) {
            for (int x = 0; x < KS; x++) {
                DU dx = *ix;
                switch (op) {
                case L_USAMPLE:
                case L_AVGPOOL: v += dx;        break;
                case L_MAXPOOL: v = MAX(dx, v); break;
                case L_MINPOOL: v = MIN(dx, v); break;
                }
                ix += C;                             /// * next cell
            }
            ix += RI;                                /// * next row
        }
        O[z0] = avg ? v / KSQ : v;
    }
}

#define SELU_L  1.0507                     /** Selu lambda */
#define SELU_LA 1.7581                     /** Selu alpha  */

__KERN__ void k_activate(
    t4_layer op, DU *I, DU *F, DU *O,      ///< func, input, filter, output tensors
    DU alpha, U64 numel                    ///< number of tensor elements
    ) {
    for (U64 j = threadIdx.x; j < numel; j += blockDim.x) {
        DU k = I[j];                                       ///< use register
        switch (op) {
        case L_RELU:
            O[j] = k > DU0
                ? (F[j]=DU1, k) : (F[j]=DU0);      break;  /// * 1|0
        case L_TANH:
            O[j] = 0.5 * (DU1 + (k=TANH(k)));              /// * scaled to [0,1)
            F[j] = DU1 - k*k;                      break;  /// * (1 - tanh^2)
        case L_SIGMOID:
            O[j] = k = SIGMOID(k);
            F[j] = k * (DU1 - k);                  break;  /// * sig*(1 - sig)
        case L_SELU: O[j] = k > DU0                        /// * selu
            ? (F[j] = SELU_L, k)
            : (F[j] = SELU_LA * EXP(k)) - SELU_LA; break;
        case L_LEAKYRL: O[j] = k > DU0
            ? (F[j] = DU1, k)
            : (F[j] = alpha) * k;                  break;
        case L_ELU:     O[j] = k > DU0
            ? (F[j] = DU1, k)
            : (F[j] = alpha * EXP(k)) - alpha;     break;
        case L_DROPOUT:
            O[j] = F[j] > alpha
            ? (F[j]=DU1, k) : (F[j]=DU0);          break;  /// * 1|0
        }
    }
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
__GPU__ Model&
Model::forward(Tensor &input) {
    Tensor &n1 = (*this)[1];    ///< reference model input layer
    if (*_trace) input.show();  /// * preview input data

    if (input.numel != n1.numel) {
        ERROR("Model::forward dataset wrong shape[%d,%d,%d,%d] != model input[[%d,%d,%d,%d]\n",
            input.N(), input.H(), input.W(), input.C(),
            n1.N(), n1.H(), n1.W(), n1.C());
        return *this;
    }
    n1 = input;               /// * copy dataset batch into the first layer [0,1)
    ///
    /// cascade execution layer by layer forward
    /// TODO: model execution becomes a superscalar pipeline
    ///
    auto trace = [](DU t, int i, Tensor &in, Tensor &out) {
        INFO("\n%6.2f:%2d> %s [%2d,%2d,%2d,%2d] Σ/n=%6.2f\tp=%6.3f => out[%2d,%2d,%2d,%2d]",
            t, i, d_nname(in.grad_fn), in.N(), in.H(), in.W(), in.C(),
            in.sum() / in.N() / in.C(), 0.001*in.parm,
            out.N(), out.H(), out.W(), out.C());
    };
    NLOG("\nModel::forward starts {");
    DU t0 = System::ms(), t1 = t0, tt;           ///< performance measurement
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (*_trace) {
            trace((tt=System::ms()) - t1, i, in, out);
            t1 = tt;
        }
        _fstep(in, out);
        out.show();
        for (int n=0; n<out.N(); n++) out._dump(out.slice(n), out.H(), out.W(), out.C());
        if (*_trace > 1) out.show();
    }
    ///
    /// collect onehot vector and hit count
    ///
    if (input.is_dataset()) {
        if (_hot) FREE(*_hot);                   /// * release if previously alloc
        _hot = &onehot((Dataset&)input);         /// * create/cache onehot vector
        _hit = hit(true);                        /// * recalc/cache hit count
    }
    NLOG("\n} Model::forward %5.2f ms\n", System::ms() - t0);

    return *this;
}
/// ========================================================================
/// private methods
///
__GPU__ void
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
        Tensor &t = *in.grad[0];
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
    default: ERROR("nn#_fstep layer=%d not supported\n", fn);
    }
}

#define TILE1    (T4_DIM_SZ)              /** 16, 1x1 conv */
#define TILE3    (T4_DIM_SZ - 3 + 1)      /** 14, 3x3 conv */
#define TILE5    (T4_DIM_SZ - 5 + 1)      /** 12, 5x5 conv */

__GPU__ int
Model::_fconv(Tensor &in, Tensor &out) {
    Tensor &tf = *in.grad[0];                             ///< filter tensor
    Tensor &tb = *in.grad[1];                             ///< bias tensor

    NN_DB(" nn#_fconv f[%d,%d,%d,%d], b[%ld]\n", tf.N(), tf.H(), tf.W(), tf.C(), tb.numel);

    const U32 N = out.N(), H = out.H(), W = out.W();      ///< outpt dimensions
    const U32 C0 = out.C(), C1 = in.C();                  ///< output, input channel deep

    dim3 blk(T4_DIM_SZ, T4_DIM_SZ, 1);                    ///< default blocks
    dim3 g1((W + TILE1 - 1) / TILE1, (H + TILE1 - 1) / TILE1, C0);
    dim3 g3((W + TILE3 - 1) / TILE3, (H + TILE3 - 1) / TILE3, C0);
    dim3 g5((W + TILE5 - 1) / TILE5, (H + TILE5 - 1) / TILE5, C0);

    for (U32 n = 0; n < N; n++) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        DU *f  = tf.data, *b = tb.data;
        U32 ks = tf.H();
        switch(ks) {                       /// * TODO: handles rectangular filters
        case 1: k_conv2d<TILE1,1><<<g1,blk>>>(d1, f, b, d0, H, W, C1); break;
        case 3: k_conv2d<TILE3,3><<<g3,blk>>>(d1, f, b, d0, H, W, C1); break;
        case 5: k_conv2d<TILE5,5><<<g5,blk>>>(d1, f, b, d0, H, W, C1); break;
        default:
            ERROR("nn#_fconv kernel_size=%d not supported\n", ks);
            return -1;
        }
        CDP_SYNC();
    }
    return 0;
}

__GPU__ int
Model::_flinear(Tensor &in, Tensor &out) {
    auto qa_calc = [&in, &out](Tensor &tw, Tensor &tb) {
        U32 N = in.N(), C1 = tw.W(), C0 = tw.H();     /// * weight dimensions
        DU *w = tw.data, *b = tb.data;
        for (U32 n = 0; n < N; n++) {                 /// * walk through batch
            DU *x = in.slice(n), *y = out.slice(n);
            for (U32 c0 = 0; c0 < C0; c0++) {
                y[c0] = b[c0];                        /// init with bias
                for (U32 c1 = 0; c1 < C1; c1++) {     /// dot product
                    y[c0] += w[C1 * c0 + c1] * x[c1]; /// Y = W @ X + B
                }
            }
        }
    };
    Tensor &tw = *in.grad[0];                         ///< weight tensor
    Tensor &tb = *in.grad[1];                         ///< bias tensor

    const U32 N  = out.N();                           ///< batch size (N1 == N0)
    const U32 C0 = tw.H(), C1 = tw.W();               ///< dense layer dims

    NN_DB(" = %d x (w[1,%d,%d,1] @ in[1,%d,%d,%d] + b[%ld])",
          N, C0, C1, in.H(), in.W(), in.C(), tb.numel);

    if (tw.numel < T4_DIM_SQ) {                       /// * threshold control
        NN_DB("*");
        qa_calc(tw, tb);                              /// * serial code
    }
    else {                                            
        FORK3(k_linear, C1, C0, N,
              in.data, out.data, tw.data, tb.data,
              in.HWC(), out.HWC());
        CDP_SYNC();
    }
    return 0;
}

__GPU__ int
Model::_factivate(Tensor &in, Tensor &out, t4_layer fn) {
    DU alpha = 0.001 * in.parm;
    FORK1(k_activate, in.numel, 
          fn, in.data, in.grad[0]->data, out.data, alpha);
    CDP_SYNC();
    return 0;
}

__GPU__ int
Model::_fpool(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W  = out.W(), H = out.H();                ///< output dimensions
    const U32 C  = out.C(), N = out.N();
    const int ks = in.parm;                             ///< kernel size

    switch(ks) {                                        /// pooling kernel size
    case 2: FORK4(k_pool<2>, fn, in.data, out.data, H, W); break;
    case 3: FORK4(k_pool<3>, fn, in.data, out.data, H, W); break;
    default:
        ERROR("nn#_fpool kernel_size=%d not supported\n", ks);
        return -1;
    }
    CDP_SYNC();
    return 0;
}

__GPU__ int
Model::_fsoftmax(Tensor &in, Tensor &out) {
    out = in;                                   /// copy content for exe calc
    out.map(EXP);                               /// *
    Tensor &t = T4(1, in.H(), in.W(), in.C());  ///< create temp tensor for calc
    DU     *d = t.data;                         ///< cached tensor data
    for (U32 n = 0; n < in.N(); n++) {          ///< loop thru mini-batch
        t.data = out.slice(n);                  /// * point to output data slice
        DU sum = t.sum();                       ///< sum(exp(xi))
        t.map(MUL, RCP(sum + DU_EPS));          /// * softmax = exp(xi)/sum(exp(xi))
    }
    t.data = d;                                 /// * restore tensor data
    FREE(t);                                    /// * release memory
    return 0;
}

__GPU__ int
Model::_flogsoftmax(Tensor &in, Tensor &out) {  /// * TODO: DCP
    out = in;                                   /// * copy in data to out
    out.map(EXP);
    Tensor &t = T4(1, in.H(), in.W(), in.C());  ///< create tmp tensor
    DU     *d = t.data;                         ///< cache tensor data
    for (U32 n = 0; n < in.N(); n++) {          /// * loop throught mini-batch
        t.data = out.slice(n);
        DU sum    = t.sum();
        DU logsum = LOG(sum > DU0 ? sum : DU_EPS);
        t -= logsum;                            ///< xi - log(sum(exp(xi)))
    }
    t.data = d;                                 /// * restore tensor data pointer
    FREE(t);                                    /// * release memory
    return 0;
}
///
///> batch norm, (batch mean=0.0, and variance=1.0)
///
__GPU__ int
Model::_fbatchnorm(Tensor &in, Tensor &out) {
    const U32 N   = out.N(), C = out.C(), H = out.H(), W = out.W(); ///< C0==C1
    const U64 HW  = (U64)H * W;                        ///< size of a page
    const U64 NHW = HW * N;                            ///< size of entire batch

    DU *w   = &in.grad[0]->data[0];                    ///< weight/gamma
    DU *b   = &in.grad[0]->data[C];                    ///< bias/beta
    DU *avg = &in.grad[1]->data[0];                    ///< mean
    DU *var = &in.grad[1]->data[C];                    ///< 1.0/(sqrt(var)+e)
    DU *xht = in.grad[3]->data;                        ///< x_hat

    #pragma unroll
    for (U32 c=0; c < C; c++) avg[c] = var[c] = DU0;   /// * zero out
    FORK4(k_batchsum, in.data, avg, HW);               /// * capture sum
    CDP_SYNC();

    #pragma unroll
    for (U32 c=0; c < C; c++) avg[c] *= DU1 / NHW;     /// * calc mean per channel
    FORK4(k_batchnvar, in.data, avg, var, HW);         /// * capture n*variance
    CDP_SYNC();

    const DU m = 0.001 * in.parm;                      ///< ETA momentum, TODO:
    #pragma unroll
    for (U32 c=0; c < C; c++) {
        var[c] = DU1 / (SQRT(var[c] / NHW) + DU_EPS);  ///< gvar = gamma/(stdvar + e)
    }
    FORK4(k_batchnorm, in.data, out.data, xht, avg, var, w, b, HW); /// * O = x_hat*gamma + beta
    CDP_SYNC();
    return 0;
}
///
///> upsampling =~ reverse pooling (calls backprop k_dpool)
///
template<int KS>                                        /// forward declare (in backprop.cu)
__KERN__ void k_dpool(t4_layer op, DU *I, DU *O, U32 H, U32 W);
__GPU__ int
Model::_fupsample(Tensor &in, Tensor &out) {
    const U32 W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const U32 C  = in.C(), N = in.N();
    const int me = (in.parm >> 8);                      ///< upsample method, TODO
    const int ks = in.parm & 0xff;                      ///< upsampling size

    switch(ks) {
    case 2: FORK4(k_dpool<2>, L_USAMPLE, out.data, in.data, H, W); break;
    case 3: FORK4(k_dpool<3>, L_USAMPLE, out.data, in.data, H, W); break;
    default:
        ERROR("model#upsample size=%d not supported\n", ks);
        return -1;
    }
    CDP_SYNC();
    
    return 0;
}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
