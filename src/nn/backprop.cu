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
    DU *I, DU *F, DU *DF, DU *DB, DU *O,   ///< input I[HxW], F,DF[KSxKS], output O[HxW]
    int H, int W, int C0                   ///< H1==H0, W1==W0, output Channels
    ) {
    __shared__ DU _I[T4_WARP_SQ];                    ///< input cache tile [16x16]
    __shared__ DU _O[T4_WARP_SQ];                    ///< output cache tile [16x16]
    
    const int KSQ= KS * KS;                          ///< save some muliplications
    const int tx = threadIdx.x, j1 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i1 = ty + blockIdx.y * TS;
    const int c1 = blockIdx.z,  C1 = gridDim.z;      ///< channel deep
    const int xy = tx + ty * T4_WARP_SZ;             ///< offset in cache window
    const int z1 = c1 + (j1 + i1 * W) * C1;          ///< input array index
    ///
    /// process z1, i.e. [TS, TS, C1] cells per kernel call
    ///
    const int i0 = i1 - INT(KS / 2);                 ///< dY coordinates
    const int j0 = j1 - INT(KS / 2);
    
    auto g = cg::this_thread_block();                ///< group all threads
    
    _I[xy] = (i1 < H && j1 < W) ? I[z1] : DU0;       ///< cached X (input) tile
    g.sync();

    for (int c0 = 0; c0 < C0; c0++) {                ///< each dY channel
        const int z0 = c0 + (j0 + i0 * W) * C0;      ///< output array index
        _O[xy] =                                     /// * cache dY (output) tile
            (i0 >= 0 && i0 < H && j0 >= 0 && j0 < W) /// * with zero padding
            ? O[z0] : DU0;                           /// * by channel
        g.sync();                                    /// * smem write barrier
        if (c1 == 0) atomicAdd(&DB[c0], _O[xy]);     /// * dB += dY
        
        const int zf = c0 + c1 * KSQ * C0;           ///< filter index F[C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                    /// * within tile [12x12]
            DU *fx = &F[zf + (KSQ - 1) * C0];        ///< F[c1,KS-1,KS-1,c0] i.e. rot180
            DU *dfx= &DF[zf], *ox = &_O[xy];         ///< DF[c1,0,0,c0], dY
            DU sum = DU0;                            ///< dX sum (TSxTS threads)
            for (int y = 0; y < KS; y++) {           /// * process one KS * KS cell
                for (int x = 0; x < KS; x++) {
                    sum     += (*fx) * ox[x];        /// * dX += F' @ dY (for each C1)
                    fx      -= C0;                   /// * walk F backward
                    atomicAdd(dfx, ox[x] * _I[xy]);  /// * dF += dY * X (TSxTS threads)
                    dfx     += C0;                   /// * DF[c1,0,1,c0]
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
    int C1, int C0, int HWC1, int HWC0
    ) {    
    const int c0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int c1 = threadIdx.y + blockIdx.y * blockDim.y;
    const int n  = blockIdx.z;
    const int cx = c1 + c0 * C1;

    if (c0 < C0 && c1 < C1) {
        DU x = I[c1 + n * HWC1], dy = O[c0 + n * HWC0];
        atomicAdd_block(&DW[cx], dy * x);           /// * dw += dY * X^t
        if (c1 == 0) {
            atomicAdd_block(&DB[c0], dy);           /// * db += dY
        }
    }
}

__KERN__ void k_dlinear_dx(
    DU *I, DU *O, DU *W,
    int C1, int C0, int HWC1, int HWC0
    ) {
    const int c1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int n  = blockIdx.z;

    if (c1 < C1) {
        DU *w = &W[c1], *y = &O[n * HWC0];          /// * dX = W^t @ dY
        DU acc = DU0;
        for (int c0 = 0; c0 < C0; c0++, w+=C1) {
            acc += (*w) * (*y++);
        }
        I[c1 + n * HWC1] = acc;
    }
}

template<int KS>                                      /// kernel size
__KERN__ void k_dpool(
    t4_layer op,
    DU *I, DU *O,                                     ///< input, output buffers
    int H, int W                                      ///< output HW (C1==C0)
    ) {
    const int HW = H * W;                             ///< HxW
    const int k0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int j0 = k0 % W;                            ///< output x dim
    const int c  = blockIdx.y, C = gridDim.y;         ///< channel deep
    const int ns = blockIdx.z * HW * C;               ///< batch slice idx
    const int z0 = c + k0 * C + ns;                   ///< output array index
    const int z1 = c + j0 * KS * C + ((k0 - j0) * C + ns) * KS * KS;
    
    if (k0 < HW && c < C) {
        const int RI = (W - 1) * KS * C;     ///< input cell row increment
        DU *ix = &I[z1], *t = ix;            /// *ix input tensor cell
        DU2 v  = (op != L_AVGPOOL) ? *ix : O[z0] / (KS * KS);
        #pragma unroll
        for (int y = 0; y < KS; y++) {       /// * handle one kernel
            for (int x = 0; x < KS; x++) {
                DU dx = *ix;
                switch (op) {
                case L_AVGPOOL: *ix = v;             break;
                case L_MAXPOOL:
                    *ix = DU0;               /// * zero out all elements
                    if (dx > v) { v = dx; t = ix; }  break;
                case L_MINPOOL:
                    *ix = DU0;
                    if (dx < v) { v = dx; t = ix; }  break;
                case L_USAMPLE: *ix = O[z0];         break;
                }
                ix += C;                    /// * next cell
            }
            ix += RI;                       /// * next input row
        }
        if (op==L_MAXPOOL || op==L_MINPOOL) *t = O[z0];    /// * update arg cell
    }
}

__KERN__ void k_dfilter(
    DU *I, DU *F, DU *O,                    ///< input, filter, output
    int HW                                  ///< H1==H0, W1==W0 (C1==C0)
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index
    
    if (i < HW && c < C) {
        I[k] = (F[k] > DU0) ? O[k] : DU0;
    }
}

///
/// backprop: Neural Network back propegation
/// Note: cascade execution layer by layer backward
///
__GPU__ Model&
Model::backprop() {
    if (_hot) return backprop(*_hot);    /// * use default one-hot vector

    ERROR("Model#backprop missing onehot vector?\n");
    return *this;
}

__GPU__ Model&
Model::backprop(Tensor &hot) {
    auto trace = [](DU t, int i, Tensor &in, Tensor &out) {
        printf("\n%6.2f:%2d> %s [%d,%d,%d,%d]\tp=%-2d <= out'Σ/n=%6.2f [%d,%d,%d,%d] ",
            t, i, d_nname(in.grad_fn),
            in.N(), in.H(), in.W(), in.C(), in.parm,
            out.sum() / out.N() / out.C(),
            out.N(), out.H(), out.W(), out.C());
    };
    TRACE1("\nModel#backprop starts");
    DU t0 = _mmu->ms(), t1 = t0, tt;            ///< performance measurement
    int x = 0;
    for (U16 i = numel - 2, j = 0; i > 0; i--, j++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (_trace) {
            trace((tt=_mmu->ms()) - t1, i, in, out);
            t1 = tt;
        }
        _bstep(in, j ? out : hot);
        if (_trace > 1) {
            debug(in, 300.0f);
        }
//        if (++x > 2) break;
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
    case L_SIGMOID: /* TODO: */ break; /// dX = dY * sigmod(X) * (1 - sigmod(X))
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
    case L_USAMPLE: _bupsample(in, out, fn); break;
    default: ERROR("Model#backprop layer=%d not supported\n", fn);
    }
}

#define TILE1    (T4_WARP_SZ)              /** 16, 1x1 conv */
#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14, 3x3 conv */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12, 5x5 conv */

__GPU__ int
Model::_bconv(Tensor &in, Tensor &out) {
    Tensor &tf = *in.grad[0], &tdf = *in.grad[2];    ///< filter tensor
    Tensor &tb = *in.grad[1], &tdb = *in.grad[3];    ///< bias tensor
    
    TRACE1(" f[%d,%d,%d,%d], b[%d]", tf.N(), tf.H(), tf.W(), tf.C(), tb.numel);
    
    const int N = in.N(), H = in.H(), W = in.W();    ///< input dimensions
    const int C1 = in.C(), C0 = out.C();            
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 g1((W + TILE1 - 1) / TILE1, (H + TILE1 - 1) / TILE1, C1);
    dim3 g3((W + TILE3 - 1) / TILE3, (H + TILE3 - 1) / TILE3, C1);
    dim3 g5((W + TILE5 - 1) / TILE5, (H + TILE5 - 1) / TILE5, C1);

    for (int n = 0; n < N; n++) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        DU *f  = tf.data,     *df = tdf.data, *db = tdb.data;
        const int ks = tf.H();                       ///< kernel size
        switch (ks) {
        case 1: k_dconv2d<TILE1,1><<<g1,blk>>>(d1, f, df, db, d0, H, W, C0); break;
        case 3: k_dconv2d<TILE3,3><<<g3,blk>>>(d1, f, df, db, d0, H, W, C0); break;
        case 5: k_dconv2d<TILE5,5><<<g5,blk>>>(d1, f, df, db, d0, H, W, C0); break;
        default: 
            ERROR("model_back#conv kernel_size %d not supported\n", ks);
            return -1;
        }
        GPU_SYNC();
    }
//     _dump_db(tdb);
//     _dump(tdf.data, tdf.H(), tdf.W(), tdf.C());
    if (_trace > 1) {
        _dump_dbdf(tdb, tdf);
    }
    return 0;
}

__GPU__ int
Model::_blinear(Tensor &in, Tensor &out) {
    Tensor &tw  = *in.grad[0];                      ///< weight tensor
    Tensor &tdw = *in.grad[2];                      ///< d_weight tensor
    Tensor &tdb = *in.grad[3];                      ///< d_bias tensor

    const int N  = out.N();                         ///< batch size (N1 == N0)
    const int H1 = in.H(), H0 = out.H();            ///< input output H
    const int C0 = tw.H(), C1 = tw.W();             ///< filter dimensions
        
    TRACE1("\n\tdw[%d,%d] += out'[%d,1] @ in^t[1,%d]", C0, C1, H0, H1);
    TRACE1("\n\tin[%d, 1]  = w^t[%d,%d] @ out'[%d,1]", H1, C1, C0, H0);

    if (tw.numel > T4_WARP_SQ) {                    /// * parallel mode
        dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
        dim3 grd(NGRID(C0, C1, N, blk));
        dim3 grx(NGRID(C1, 1, N, blk));

        k_dlinear_dwdb<<<grd, blk>>>(               /// * update dB, dW
            in.data, out.data, tdw.data, tdb.data,
            C1, C0, in.HWC(), out.HWC());
        k_dlinear_dx<<<grx, blk>>>(                 /// * update dX (in parallel)
            in.data, out.data, tw.data,
            C1, C0, in.HWC(), out.HWC());
        GPU_SYNC();
    }
    else {                                          /// * serial mode (for validation)
        TRACE1("*");
        for (int n = 0; n < N; n++) {
            DU *x = in.slice(n), *y = out.slice(n), *dw = tdw.data;
            for (int c0 = 0; c0 < C0; c0++) {       /// W[C0,C1]
                DU yi = y[c0];
                tdb[c0] += yi;                      /// * db += dY
                for (int c1 =0; c1 < C1; c1++) {
                    *dw++ += yi * x[c1];            /// * dw += dY @ X^t
                }
            }
            DU *w = tw.data;
            for (int c1 = 0; c1 < C1; c1++) {       /// * dX = w^t @ dY
                DU sum = DU0;
                for (int c0 = 0; c0 < C0; c0++) {
                    sum += w[c1 + c0 * C1] * y[c0];
                }
                x[c1] = sum;
            }
        }
    }
    // _dump(in.data, in.H(), in.W(), in.C());
    if (_trace > 1) {
         _dump_db(tdb);
         _dump_dw(tdw, false);
    }
    return 0;
}

__GPU__ int
Model::_bfilter(Tensor &in, Tensor &msk, Tensor &out) {
    const int W = out.W(), HW = out.H() * W;
    
    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((HW + blk.x -1) / blk.x, out.C(), out.N());

    k_dfilter<<<grd,blk>>>(in.data, msk.data, out.data, HW);
    GPU_SYNC();

    return 0;
}

__GPU__ int
Model::_bpool(Tensor &in, Tensor &out, t4_layer fn) {
    const int W = out.W(), H = out.H();   ///< output dimensions
    
    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((H * W + blk.x - 1) / blk.x, out.C(), out.N());

    const int ks = in.parm;               ///< kernel size
    switch(ks) {                           
    case 2: k_dpool<2><<<grd,blk>>>(fn, in.data, out.data, H, W); break;
    case 3: k_dpool<3><<<grd,blk>>>(fn, in.data, out.data, H, W); break;
    default:
        ERROR("model#pooling kernel_size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();
    
    return 0;
}
///
///> upsampling =~ reverse pooling (calls forward k_pool)
///
template<int KS>                                        /// forward declare (in forward.cu)
__KERN__ void  k_pool(t4_layer op, DU *I, DU *O, int H, int W);
__GPU__ int
Model::_bupsample(Tensor &in, Tensor &out, t4_layer fn) {
    const int W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const int me = (in.parm >> 8);                      ///< upsample method, TODO
    const int ks = (in.parm & 0xff);                    ///< kernel size
    
    dim3 blk(T4_WARP_SQ, 1, 1);                         ///< default blocks
    dim3 grd((H * W + blk.x - 1) / blk.x, in.C(), in.N());
    
    switch(ks) {                                        /// by kernel size
    case 2: k_pool<2><<<grd,blk>>>(fn, out.data, in.data, H, W); break;
    case 3: k_pool<3><<<grd,blk>>>(fn, out.data, in.data, H, W); break;
    default:
        ERROR("model#upsample size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();
    
    _dump(out.data, out.H(), out.W(), out.C());
    _dump(in.data, in.H(), in.W(), in.C());
    return 0;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
