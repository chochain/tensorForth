/** -*- c++ -*-
 * @file
 * @brief Model class - backward propagation implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if (T4_DO_OBJ && T4_DO_NN)
///
/// convolution filter derivatives
/// TODO: stride, dilation, [C1]NCHW filter
///
template<int TS, int KS>    ///> tile size, kernel size
__KERN__ void k_dconv2d(
    DU *I, DU *F, DU *DF, DU *DB, DU *O,   ///< input I[HxW], F,DF[KSxKS], output O[HxW]
    int H, int W, int C0, bool train       ///< H1==H0, W1==W0, output Channels
    ) {
    __shared__ DU _I[T4_WARP_SQ];                    ///< input cache tile [16x16]
    __shared__ DU _O[T4_WARP_SQ];                    ///< output cache tile [16x16]

    const U32 KSQ= KS * KS;                          ///< save some muliplications
    const U32 tx = threadIdx.x, j1 = tx + blockIdx.x * TS;
    const U32 ty = threadIdx.y, i1 = ty + blockIdx.y * TS;
    const U32 c1 = blockIdx.z,  C1 = gridDim.z;      ///< channel deep
    const U64 z1 = ((U64)W * i1 + j1) * C1 + c1;     ///< input array index
    const U64 xy = (U64)T4_WARP_SZ * ty + tx;        ///< offset in cache window
    ///
    /// process z1, i.e. [TS, TS, C1] cells per kernel call
    ///
    const int i0 = i1 - INT(KS / 2);                 ///< dY coordinates
    const int j0 = j1 - INT(KS / 2);

    auto g = cg::this_thread_block();                ///< group all threads

    _I[xy] = (i1 < H && j1 < W) ? I[z1] : DU0;       ///< cached X (input) tile
    g.sync();

    for (U32 c0 = 0; c0 < C0; c0++) {                ///< each dY channel
        const U64 z0 = ((U64)W * i0 + j0) * C0 + c0; ///< output array index
        _O[xy] =                                     /// * cache dY (output) tile
            (i0 >= 0 && i0 < H && j0 >= 0 && j0 < W) /// * with zero padding
            ? O[z0] : DU0;                           /// * by channel
        g.sync();                                    /// * smem write barrier
        if (train && c1 == 0) {
            atomicAdd(&DB[c0], _O[xy]);              /// * dB += dY
        }
        const U64 zf = (U64)C0 * KSQ * c1 + c0;      ///< filter index F[C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                    /// * within tile [12x12]
            DU *fx = &F[zf + (KSQ - 1) * C0];        ///< F[c1,KS-1,KS-1,c0] i.e. rot180
            DU *dfx= &DF[zf], *ox = &_O[xy];         ///< DF[c1,0,0,c0], dY
            DU sum = DU0;                            ///< dX sum (TSxTS threads)
            for (U32 y = 0; y < KS; y++) {           /// * process one KS * KS cell
                for (U32 x = 0; x < KS; x++) {
                    sum += (*fx) * ox[x];            /// * dX += F' @ dY (for each C1)
                    fx  -= C0;                       /// * walk F backward
                    if (!train) continue;
                    atomicAdd(dfx, ox[x] * _I[xy]);  /// * dF += dY * X (TSxTS threads)
                    dfx += C0;                       /// * DF[c1,0,1,c0]
                }
                ox += T4_WARP_SZ;
            }
            if (i1 < H && j1 < W) {                  /// * update input matrix
                if (c0 == 0) I[z1] = sum;            /// * update I (per C1)
                else         I[z1] += sum;
            }
        }
        g.sync();                                    /// * d read barrier
    }
}

__KERN__ void k_dlinear_dwdb(
    DU *I, DU *O, DU *DW, DU *DB,
    U64 HWC1, U64 HWC0, U32 C1, U32 C0
    ) {
    const U64 c1 = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    const U64 c0 = (U64)blockIdx.y * blockDim.y + threadIdx.y; 
    const U64 cx = c0 * C1 + c1;
    const U32 n  = blockIdx.z;

    if (c0 < C0 && c1 < C1) {                          /// * TODO: shuffle-sum
        DU dy = O[HWC0 * n + c0];
        DU x  = I[HWC1 * n + c1];
        atomicAdd(&DW[cx], dy * x);                    /// * dw += dY @ X^t
        if (c1 == 0) atomicAdd(&DB[c0], dy);           /// * db += dY
    }
}

__KERN__ void k_dlinear_dx(
    DU *I, DU *O, DU *W,
    U64 HWC1, U64 HWC0, U32 C1, U32 C0
    ) {
    const U64 c1 = (U64)blockIdx.x * blockDim.x + threadIdx.x;
    const U64 c0 = (U64)blockIdx.y * blockDim.y + threadIdx.y;
    const U32 n  = blockIdx.z;
    const U64 cx = (U64)C1 * c0 + c1;

    if (c0 < C0 && c1 < C1) {                          /// * TODO: shuffle-sum
        DU dy = O[HWC0 * n + c0];
        DU *x = &I[HWC1 * n + c1];                     ///< pointer to X
        atomicAdd(x, W[cx] * dy);                      /// * dX = W^t * dY
    }
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
    const U64 ns = HW * blockIdx.z * C;               ///< batch slice idx
    const U64 z0 = (U64)C * k0 + ns + c;              ///< output array index
    const U64 z1 = (U64)C * KS * j0 + KSQ * ((k0 - j0) * C + ns) + c;

    if (k0 < HW && c < C) {
        const U64 RI = (U64)KS * C * (W - 1);         ///< input cell row increment
        DU *ix = &I[z1], *t = ix;                     /// *ix input tensor cell
        DU2 v  = (op != L_AVGPOOL) ? *ix : O[z0] / KSQ;
        #pragma unroll
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

__KERN__ void k_dactivate(
    DU *I, DU *F, DU *O,                   ///< input, filter, output
    U64 numel                              ///< tensor element count
    ) {
    const U64 j = (U64)blockIdx.x * blockDim.x + threadIdx.x;   ///< element index

    if (j < numel) I[j] = O[j] * F[j];     /// * Harmand product
}

__KERN__ void k_dbatchnorm_1(
    DU *I, DU *O, DU *X,                   ///< input, output, x_hat tensors
    DU *sum, DU *g_var,                    ///< sum(x_hat), gamma/(stdvar+e)
    U64 HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const U64 j  = (U64)threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const U32 c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const U64 ns = HW * blockIdx.z * C;                    ///< batch slice index
    const U64 k  = (U64)C * j + ns + c;                    ///< output tensor index
    const DU  _N = 1.0 / gridDim.z;                        ///< 1.0/HWN

    if (j < HW) {
        I[k] = (O[k] - sum[c] * _N) * g_var[c];            /// * dX = g_var * (dout - sum(dout) / N)
        O[k] *= X[k];                                      /// * dout * x_hat
    }
}
__KERN__ void k_dbatchnorm_2(
    DU *I, DU *X, DU *sum,                 ///< input, x_hat
    U64 HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const U64 j  = (U64)blockIdx.x * blockDim.x + threadIdx.x;  ///< element index
    const U32 c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const U64 ns = HW * C * blockIdx.z;                    ///< batch slice index
    const U64 k  = (U64)C * j + ns + c;                    ///< output tensor index

    if (j < HW) I[k] -= X[k] * sum[c];
}
///
/// backprop: Neural Network back propegation
/// Note: cascade execution layer by layer backward
///
__GPU__ Model&
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

__GPU__ Model&
Model::backprop() {
    if (_hot) return backprop(*_hot);            /// * use default one-hot vector

    ERROR("Model#backprop missing onehot vector?\n");
    return *this;
}

__GPU__ Model&
Model::backprop(Tensor &tgt) {
    auto trace = [](DU t, int i, Tensor &in, Tensor &out) {
        printf("\n%6.2f:%2d> %s [%d,%d,%d,%d]\tp=%-2d <= out'Σ/n=%6.2f [%d,%d,%d,%d] ",
            t, i, d_nname(in.grad_fn),
            in.N(), in.H(), in.W(), in.C(), in.parm,
            out.sum() / out.N() / out.C(),
            out.N(), out.H(), out.W(), out.C());
    };
    if (_bloss(tgt)) return *this;                 /// * pre-calculate dLoss
    
    MM_DB("\nModel#backprop starts");
    DU  t0 = System::ms(), t1 = t0, tt;                   ///< performance measurement
    for (int i = numel - 2, j = 0; i > 0; i--, j++) {     /// numel=number of layers
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (_trace) {
            trace((tt=System::ms()) - t1, i, in, out); t1 = tt;
            _bstep(in, out);
            in.show();
        }
        else _bstep(in, out);
    }
    MM_DB("\nModel::backprop %5.2f ms\n", System::ms() - t0);
    return *this;
}
/// ========================================================================
/// private methods
///
__GPU__ int
Model::_bloss(Tensor &tgt) {                     ///> pre-calc dLoss
    Tensor &out = (*this)[-1];                   ///< output layer, used as dLoss
    if (tgt.numel != out.numel) {                /// * check dimensions of target vector
        ERROR("\nERROR: Onehot wrong shape[%d,%d,%d,%d] != [%d,%d,%d,%d]\n",
              tgt.N(), tgt.H(), tgt.W(), tgt.C(),
              out.N(), out.H(), out.W(), out.C());
        return 1;
    }
    MM_DB("\nModel#backprop: input dimensions OK, calculate dLoss");
    t4_layer fn = (*this)[-2].grad_fn;           ///< final activation layer
    switch (fn) {
    case L_SIGMOID:                              /// * sigmoid + BCE
    case L_SOFTMAX:                              /// * softmax + CE
    case L_LOGSMAX: out -= tgt;  break;          /// * log-softmax + NLL
    default:        out  = tgt;  break;          /// * pre-calc dLoss (pass thru)
    }
    if (_trace) out.show();                      /// * display loss if trace on

    return 0;
}

__GPU__ void
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
    default: ERROR("Model#backprop layer=%d not supported\n", fn);
    }
}

#define TILE1    (T4_WARP_SZ)              /** 16, 1x1 conv */
#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14, 3x3 conv */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12, 5x5 conv */

__GPU__ int
Model::_bconv(Tensor &in, Tensor &out) {
    Tensor &w = *in.grad[0], &dw = *in.grad[2];      ///< filter tensor
    Tensor &b = *in.grad[1], &db = *in.grad[3];      ///< bias tensor

    MM_DB(" f[%d,%d,%d,%d], b[%ld]", w.N(), w.H(), w.W(), w.C(), b.numel);

    const U32 N = in.N(), H = in.H(), W = in.W();    ///< input dimensions
    const U32 C1 = in.C(), C0 = out.C();

    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 g1((W + TILE1 - 1) / TILE1, (H + TILE1 - 1) / TILE1, C1);
    dim3 g3((W + TILE3 - 1) / TILE3, (H + TILE3 - 1) / TILE3, C1);
    dim3 g5((W + TILE5 - 1) / TILE5, (H + TILE5 - 1) / TILE5, C1);

    for (U32 n = 0; n < N; n++) {                   ///< accumulative over N samples
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        const U32 ks = w.H();                       ///< kernel size
        switch (ks) {
        case 1: k_dconv2d<TILE1,1><<<g1,blk,0,cudaStreamTailLaunch>>>(
                    d1, w.data, dw.data, db.data, d0, H, W, C0, train); break;
        case 3: k_dconv2d<TILE3,3><<<g3,blk,0,cudaStreamTailLaunch>>>(
                    d1, w.data, dw.data, db.data, d0, H, W, C0, train); break;
        case 5: k_dconv2d<TILE5,5><<<g5,blk,0,cudaStreamTailLaunch>>>(
                    d1, w.data, dw.data, db.data, d0, H, W, C0, train); break;
        default:
            ERROR("model_back#conv kernel_size %d not supported\n", ks);
            return -1;
        }
        // GPU_SYNC();
    }
    if (_trace > 1) _dump_dbdf(db, dw);
    return 0;
}

__GPU__ int
Model::_blinear(Tensor &in, Tensor &out) {
    auto qa_calc = [&in, &out](Tensor &w, Tensor &dw, Tensor &db, bool train) {
        const U32 N = in.N(), C1 = w.W(), C0 = w.H(); /// * weight dimensions
        for (U32 n = 0; n < N; n++) {               ///< acc over N samples
            DU *x = in.slice(n), *y = out.slice(n);
            if (train) {
                DU *dp = dw.data;
                for (U32 c0 = 0; c0 < C0; c0++) {   /// W[C0,C1]
                    DU yi = y[c0];
                    db[c0] += yi;                   /// * db += dY
                    for (U32 c1 =0; c1 < C1; c1++) {
                        *dp++ += yi * x[c1];        /// * dw += dY @ X^t
                    }
                }
            }
            DU *wd = w.data;
            for (U32 c1 = 0; c1 < C1; c1++) {       /// * dX = w^t @ dY
                DU sum = DU0;
                for (U32 c0 = 0; c0 < C0; c0++) {
                    sum += wd[c1 + c0 * C1] * y[c0];
                }
                x[c1] = sum;
            }
        }
    };                    
    Tensor &w  = *in.grad[0];                       ///< weight tensor
    Tensor &dw = *in.grad[2];                       ///< d_weight tensor
    Tensor &db = *in.grad[3];                       ///< d_bias tensor

    const U32 N  = out.N();                         ///< batch size (N1 == N0)
    const U32 C0 = w.H(), C1 = w.W();               ///< weight tensor dimensions
    const U64 E1 = in.HWC(), E0 = out.HWC();        ///< input, output element count

    MM_DB("\n\tdw[%d,%d] += out'[%ld,1] @ in^t[1,%ld]", C0, C1, E0, E1);
    MM_DB("\n\tin[%ld, 1] = w^t[%d,%d] @ out'[%ld,1]", E1, C1, C0, E0);

    if (w.numel < T4_WARP_SQ) {                     /// * threshold control
        MM_DB("*");
        qa_calc(w, dw, db, train);                  /// * serial mode (validation)
    }
    else {
        if (train) {
            FORK3(k_dlinear_dwdb, C1, C0, N,        /// * update dB, dW
                  in.data, out.data,
                  dw.data, db.data, E1, E0);
            // GPU_SYNC();
        }
        /// barrier for X (because we did N samples in one grid)
        in.map(FILL, DU0);                          /// * zero out dX
        FORK3(k_dlinear_dx, C1, C0, N,              /// * update dX
              in.data, out.data, w.data, E1, E0);
    }
    if (train && _trace > 1) {
         _dump_db(db);
         _dump_dw(dw, true);
    }
    return 0;
}

__GPU__ int
Model::_bactivate(Tensor &in, Tensor &out) {
    Tensor::ten_op(MUL, out, *in.grad[0], in);     /// * in = msk * out
    return 0;
}

__GPU__ int
Model::_bpool(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W = out.W(), H = out.H();           ///< output dimensions
    const U32 C = out.C(), N = out.N();
    const int ks = in.parm;                       ///< kernel size
    switch(ks) {
    case 2: FORK4(k_dpool<2>, fn, in.data, out.data, H, W); break;
    case 3: FORK4(k_dpool<3>, fn, in.data, out.data, H, W); break;
    default:
        ERROR("model#pooling kernel_size=%d not supported\n", ks);
        return -1;
    }
    // GPU_SYNC();
    return 0;
}
///
///> upsampling =~ reverse pooling (calls forward k_pool)
///
template<int KS>                                        /// forward declare (in forward.cu)
__KERN__ void k_pool(t4_layer op, DU *I, DU *O, U32 H, U32 W);
__GPU__ int
Model::_bupsample(Tensor &in, Tensor &out, t4_layer fn) {
    const U32 W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const U32 C  = in.C(), N = in.N();
    const int me = (in.parm >> 8);                      ///< upsample method, TODO
    const int ks = (in.parm & 0xff);                    ///< kernel size

    switch(ks) {                                        /// by kernel size
    case 2: FORK4(k_pool<2>, fn, out.data, in.data, H, W); break;
    case 3: FORK4(k_pool<3>, fn, out.data, in.data, H, W); break;
    default:
        ERROR("model#upsample size=%d not supported\n", ks);
        return -1;
    }
    // GPU_SYNC();
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
extern __KERN__ void k_sum(DU *I, DU *sum, U64 HW);
__GPU__ int
Model::_bbatchnorm(Tensor &in, Tensor &out) {
    const U32 C = out.C(), N = out.N(), W = out.W(), H = out.H();   ///< C0==C1, N1=N0
    const U64 HW = (U64)W * H;

    DU *w   = &in.grad[0]->data[0];                    ///< weight/gamma (scale)
    DU *dw  = &in.grad[2]->data[0];                    ///< d_gamma
    DU *db  = &in.grad[2]->data[C];                    ///< d_beta
    DU *sum = &in.grad[1]->data[0];                    ///< batch sum
    DU *var = &in.grad[1]->data[C];                    ///< batch 1.0 / (var+e)^0.5
    DU *xht = in.grad[3]->data;                        ///< x_hat

    for (U32 c=0; c < C; c++) sum[c] = DU0;            /// * zero
    FORK4(k_sum, out.data, sum, HW);                   /// * capture out sum(dout)     
    // GPU_SYNC();
    
    for (U32 c=0; c < C; c++) {
        if (train) db[c] += (sum[c] /= HW);            /// * collect dbeta = sum(dout) (/ HW?)
        var[c] *= w[c];                                /// * var <= gamma * ivar
    }
    FORK4(k_dbatchnorm_1,                              /// * dX = gamma*ivar*(dout - sum(dout)/N)
        in.data, out.data, xht, sum, var, HW);         /// * also, dout *= x_hat
    // GPU_SYNC();
    
    for (U32 c=0; c < C; c++) sum[c] = DU0;            /// * zero
    FORK4(k_sum, out.data, sum, HW);                   /// * capture sum(dout * x_hat)
    // GPU_SYNC();

    for (U32 c=0; c < C; c++) {
        if (train) dw[c]  += (sum[c] /= HW);           /// * collect dgamma = sum(dout * x_hat)( / HW?)
        sum[c] *= var[c] / N;                          /// * scale sum
    }
    FORK4(k_dbatchnorm_2, in.data, xht, sum, HW);      /// * dX -= gamma*ivar*x_hat*sum(dout * x_hat) / N
    // GPU_SYNC();
    
    return 0;
}

#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
