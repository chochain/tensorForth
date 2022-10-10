/** -*- c++ -*-
 * @file
 * @brief Model class - backward propagation implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
///
/// convolution filter derivatives
/// TODO: stride, dilation, [C1]NCHW filter
///
template<int TS, int KS>         ///> tile size, kernel size
__KERN__ void k_dconv2d(
    DU *I, DU *F, DU *DF, DU *O, ///> input I[HxW], F,DF[KSxKS], output O[HxW]
    int H, int W, int C0         ///< H1==H0, W1==W0, output Channels
    ) {
    __shared__ DU _I[T4_WARP_SQ];                    ///< input cache [16x16]
    __shared__ DU _O[T4_WARP_SQ];                    ///< output cache [16x16]
    __shared__ DU _D[TS * TS * KS * KS];             ///< df cache 
    
    const int tx = threadIdx.x, j1 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i1 = ty + blockIdx.y * TS;
    const int c1 = blockIdx.z,  C1 = gridDim.z;      ///< channel deep
    const int t0 = tx + ty * T4_WARP_SZ;             ///< offset in cache window
    const int z1 = c1 + (j1 + i1 * W) * C1;          ///< input array index
    ///
    /// process z1, i.e. [TS, TS, C1] cells per kernel call
    ///
    const int i0 = i1 - int(KS / 2);                 ///< dY coordinates
    const int j0 = j1 - int(KS / 2);

    auto g = cg::this_thread_block();                ///< group all threads
    _I[t0] = (i1 < H && j1 < W) ? I[z1] : DU0;       ///< cached X (input) tile
    g.sync();

    for (int c0 = 0; c0 < C0; c0++) {                ///< each dY channel
        const int z0 = c0 + (j0 + i0 * W) * C0;
        _O[t0] =                                     /// * cache dY (output) tile
            (i0 >= 0 && i0 < H && j0 >= 0 && j0 < W) /// * with zero padding
            ? O[z0] : DU0;                           /// * by channel
        g.sync();                                    /// * smem write barrier

        const int zf = c0 + c1 * KS * KS * C0;       ///< filter index F[C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                    /// * within tile [12x12]
            DU *fx = &F[zf + (KS * KS - 1) * C0];    ///< F[KS-1,KS-1] i.e. rot180
            DU *dx = &_D[(tx + ty * TS) * KS * KS];  ///< df cache index
            DU sum = DU0;
            for (int y = 0; y < KS; y++) {           /// * process one KS * KS cell
                int k = y * T4_WARP_SZ + t0;         ///< k = filter.off + tile.off
                for (int x = 0; x < KS; x++, k++) {
                    sum     += (*fx) * _O[k];        /// * dX += F @ dY
                    *(dx++)  = _O[k] * _I[k];        /// * collect dY * X in tile
                    fx      -= C0;                   /// * walk F backward
                }
            }
            if (i1 < H && j1 < W) {                  /// * update input matrix
                if (c0 == 0) I[z1] = sum;
                else         I[z1] += sum;
            }
        }
        g.sync();                                    /// * d read barrier

        if (tx < KS && ty < KS) {
            int k = tx + ty * KS;
            for (int i = 0; i < TS * TS; i++) {
                DF[zf + k * C0] += _D[i * KS * KS + k];
            }
        }
        g.sync();
    }
}

template<int KS>                            /// kernel size
__KERN__ void k_dpool(
    DU *I, DU *O,                           ///< input, output buffers
    int H, int W,                           ///< output HW (C1==C0)
    t4_layer op
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = threadIdx.z, C = blockDim.z;        ///< channel deep
    const int n  = blockIdx.z;                         ///< batch slice id (N1==N0)
    const int ns = n * H * W * C;                      ///< slice size
    const int z0 = c + (j0 + i0 * W) * C + ns;         ///< output tensor index
    const int z1 = c + (j0 + i0 * W * KS) * KS * C + ns * KS * KS;
    
    if (i0 < H && j0 < W && c < C) {
        DU *ix = &I[z1], *t = ix;
        DU2 v  = (op != L_AVGPOOL) ? *ix : O[z0] / (KS * KS);
        for (int y = 0; y < KS; y++) {      /// * handle one kernel
            for (int x = 0; x < KS; x++) {
                DU dx = *ix;
                switch (op) {
                case L_MAXPOOL:
                    *ix = DU0;              /// * zero out all elements
                    if (dx > v) { v = dx; t = ix; }  break;
                case L_AVGPOOL: *ix = v;             break;
                case L_MINPOOL:
                    *ix = DU0;
                    if (dx < v) { v = dx; t = ix; }  break;
                }
                ix += C;
            }
            ix += (W - 1) * KS * C;
        }
        if (op != L_AVGPOOL) *t = O[z0];   /// * update arg cell
    }
}

__KERN__ void k_dfilter(
    DU *I, DU *F, DU *O,                   ///< input, filter, output
    int H, int W                           ///< H1==H0, W1==W0 (C1==C0)
    ) {
    const int j1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i1 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = threadIdx.z, C = blockDim.z;        ///< channel deep
    const int ns = blockIdx.z * H * W * C;             ///< batch slice idx
    const int z1 = c + (i1 + j1 * W) * C + ns;
    
    if (i1 < H && j1 < W && c < C) {
        I[z1] = (F[z1] > DU0) ? O[z1] : DU0;
    }
}

///
/// backprop: Neural Network back propegation
/// Note: cascade execution layer by layer backward
///
__GPU__ Model&
Model::backprop() {
    if (_dset) return backprop(*_hot);           /// * use default one-hot vector

    ERROR("Model#backprop missing onehot vector?\n");
    return *this;
}

__GPU__ Model&
Model::backprop(Tensor &hot) {
    DU t0 = _mmu->ms();                          ///< performance measurement
    auto trace = [](int i, Tensor &in, Tensor &out) {
        printf("\n%2d> %s [%d,%d,%d,%d]\tp=%-2d <= out'Σ/n=%6.2f [%d,%d,%d,%d] ",
            i, d_nname(in.grad_fn),
            in.N(), in.H(), in.W(), in.C(), in.parm,
            out.sum() / out.N() / out.C(),
            out.N(), out.H(), out.W(), out.C());
    };
    (*this)[-1] = hot;  /// softmax + CE : copy one-hot vector to model output
                        /// TODO: logsoftmax + NLL
    TRACE1("\nModel#backprop starts");
    for (U16 i = numel - 2; i > 0; i--) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (_trace > 0) trace(i, in, out);
        _bstep(in, out);
//        debug(in, 300.0f);
    }
    TRACE1("\nModel::backprop %5.2f ms\n", _mmu->ms() - t0);
    return *this;
}
/// ========================================================================
/// private methods 
///
__GPU__ void
Model::_bstep(Tensor &in, Tensor &out) {
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                     ///< layer function
    switch(fn) {
    case L_CONV:    _bconv(in, out);       break; /// * convolution
    case L_LINEAR:  _blinear(in, out);     break; /// * out = w @ in + b
    case L_FLATTEN: in = out;              break; /// * pass dY to X
    case L_RELU:    _bfilter(in, in, out); break;
    case L_TANH:    /* TODO: */ break;
    case L_SIGMOID: /* TODO: */ break;
    case L_SOFTMAX: /* softmax + CrossEntropy derivative, out = one-hot */
    case L_LOGSMAX: /* log-softmax + NLL      derivative, out = one-hot */
        in -= out;  /* softmax:    Xi = Yi - Li     */
                    /* logsoftmax: Xi = Yi - Li * p */
        break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: _bpool(in, out, fn); break;
    case L_DROPOUT: {
        Tensor &msk = *in.grad[0];             ///< dropout mask
        _bfilter(in, msk, out);
    } break;
    default: ERROR("Model#backprop layer=%d not supported\n", fn);
    }
}

#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14 */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12 */

__GPU__ int
Model::_bconv(Tensor &in, Tensor &out) {
    Tensor &tf = *in.grad[0], &tdf = *in.grad[2];    ///< filter tensor
    Tensor &tb = *in.grad[1], &tdb = *in.grad[3];    ///< bias tensor
    
    TRACE1(" f[%d,%d,%d,%d], b[%d]", tf.N(), tf.H(), tf.W(), tf.C(), tb.numel);
    
    const int N = in.N(), H = in.H(), W = in.W();    ///< input dimensions
    const int C1 = in.C(), C0 = out.C();            
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 g3((W + TILE3 - 1) / TILE3, (H + TILE3 - 1) / TILE3, C1);
    dim3 g5((W + TILE5 - 1) / TILE5, (H + TILE5 - 1) / TILE5, C1);

    for (int n = 0; n < N; n++) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        DU *f  = tf.data,     *df = tdf.data;
        const int ks = tf.H();                       ///< kernel size
        switch (ks) {
        case 3: k_dconv2d<TILE3,3><<<g3,blk>>>(d1, f, df, d0, H, W, C0); break;
        case 5: k_dconv2d<TILE5,5><<<g5,blk>>>(d1, f, df, d0, H, W, C0); break;
        default: 
            ERROR("model_back#conv kernel_size %d not supported\n", ks);
            return -1;
        }
    }
    GPU_SYNC();
    
    /// accumulate dB = sum(dY), TODO: CDP
    DU *db = tdb.data;
    for (int c0 = 0; c0 < C0; c0++, db++) {
        DU *ox = out.data + c0;
        for (int k = 0; k < H * W; k++, ox+=C0) *db += *ox;
    }
    if (_trace > 1)  _dump_dbdf(tdb, tdf);
    
    return 0;
}

__GPU__ int
Model::_blinear(Tensor &in, Tensor &out) {
    Tensor &tw  = *in.grad[0];            ///< weight tensor
    Tensor &tdw = *in.grad[2];            ///< d_weight tensor
    Tensor &tdb = *in.grad[3];            ///< d_bias tensor

    const int N  = out.N();               ///< batch size (N1 == N0)
    const int C0 = tw.H(), C1 = tw.W();   ///< filter dimensions
        
    TRACE1("\n\tdw[%d,%d] += out'[%d,1] @ in^t[1,%d]", C0, C1, out.H(), in.H());
    TRACE1("\n\tin[%d, 1]  = w^t[%d,%d] @ out'[%d,1]", in.H(), C1, C0, out.H());

    for (int n = 0; n < N; n++) {                                  ///* TODO: kernel version
        DU *x = in.slice(n), *y = out.slice(n), *dw = tdw.data;    /// * dw[C0xC1]
        for (int i = 0; i < C0; i++) {
            DU yi = y[i];
            tdb[i] += yi;                                          /// * db += dY
            for (int j =0; j < C1; j++) {
                *dw++ += yi * x[j];                                /// * dw += dY @ X^t
            }
        }
        for (int i = 0; i < C1; i++) {                             /// * dX = w^t @ dY
            DU *w =&tw.data[i], sum = DU0;
            for (int j = 0; j < C0; j++) {
                sum += *w * y[j];
                w   += C1;
            }
            x[i] = sum;
        }
    }
    if (_trace > 1) {
        _dump_db(tdb);
        _dump_dw(tdw, false);
    }
    return 0;
}

__GPU__ int
Model::_bfilter(Tensor &in, Tensor &tm, Tensor &out) {
    const int H = in.H(), W = in.W();
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, in.C());
    dim3 grd(NGRID(W, H, in.N(), blk));

    k_dfilter<<<grd,blk>>>(in.data, tm.data, out.data, H, W);
    GPU_SYNC();

    return 0;
}

__GPU__ int
Model::_bpool(Tensor &in, Tensor &out, t4_layer fn) {
    const int W = out.W(), H = out.H(); ///< output dimensions
    const int N = out.N(), C0 = out.C(); ///< batch and channel size
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, C0);
    dim3 grd(NGRID(W, H, N, blk));

    const int ks = in.parm;               ///< kernel size
    switch(ks) {                           
    case 0x2: k_dpool<2><<<grd,blk>>>(in.data, out.data, H, W, fn); break;
    case 0x3: k_dpool<3><<<grd,blk>>>(in.data, out.data, H, W, fn); break;
    default:
        ERROR("model#pooling kernel_size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();
    
    return 0;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
