/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"
#include "dataset.h"

#if T4_ENABLE_OBJ
///
/// convolution filter
/// TODO: stride, dilation, [C1]NCHW filter
///
template<int TS, int KS>         ///> tile size, kernel size
__KERN__ void k_conv2d(
    DU *I, DU *F, DU *B, DU *O,  ///> input I[HxW], F[KxK] kernel, B[C] bias, output O[HxW]
    int H, int W, int C1         ///< (H0==H1, W0==W1), input channels
    ) {
    __shared__ DU it[T4_WARP_SQ];                    ///< shared memory [16x16]
    
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * TS;
    const int C0 = gridDim.z,   c0 = blockIdx.z;     ///< output channels
    const int z0 = c0 + (j0 + i0 * W) * C0;          ///< output array index
    const int zt = tx + ty * T4_WARP_SZ;             ///< tile index
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const int i1 = i0 - int(KS / 2);                 ///< input coordinates
    const int j1 = j0 - int(KS / 2);

    auto g = cg::this_thread_block();                ///< all threads of block

    for (int c1 = 0; c1 < C1; c1++) {                ///< each input channel
        it[zt] =                                     /// * cache input data
            (i1 >= 0 && i1 < H && j1 >= 0 && j1 < W) /// * with zero padding
            ? I[c1 + (j1 + i1 * W) * C1] : DU0;      /// * by channel
        g.sync();                                    /// * smem write barrier
        ///
        /// Y = sum(W * X)
        /// TODO: cache F
        ///
        const int zf = (c1 + c0 * C1) * KS * KS;     ///< filter index
        if (tx < TS && ty < TS) {                    /// * each tile
            DU sum = DU0;
            DU *fx = &F[zf];                         /// * filter[0] ptr
            DU *ix = &it[zt];                        /// * tile[tx,ty]
            for (int y = 0; y < KS; y++) {           /// * each filter
                for (int x = 0; x < KS; x++) {
                    sum += (*fx) * ix[x];            /// Y += W * X
                    fx += C1;                        /// * next filter cell
                }
                ix += T4_WARP_SZ;                    /// next row of tile
            }
            if (i0 < W && j0 < H) {
                if (c1==0) O[z0] = sum + B[c0];      /// * O[ijc] with bias
                else       O[z0] += sum;
            }
        }
        g.sync();                          /// * d read barrier
    }
}

template<int KS>                           /// kernel size
__KERN__ void k_pool(
    DU *I, DU *O,                          ///< input, output buffers
    int H0, int W0,                        ///< output HW (C0==C1)
    t4_layer op                            ///< pooling ops
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = blockIdx.z, C = gridDim.z;
    const int z0 = j0 + i0 * W0;            ///< output array index
    const int z1 = (j0 + i0 * W0 * KS) * KS;///< input array index
    auto g = cg::this_thread_block();
    
    if (i0 < H0 && j0 < W0 && c < C) {
        DU *ix = &I[c + z1 * C];
        DU2 v  = op==L_AVGPOOL ? DU0 : *ix;
        for (int y = 0; y < KS; y++) {
            for (int x = 0; x < KS; x++) {
                DU dx = *ix;
                switch (op) {
                case L_MAXPOOL: v = MAX(dx, v); break;
                case L_AVGPOOL: v += dx;        break;
                case L_MINPOOL: v = MIN(dx, v); break;
                }
                ix += C;
            }
            ix += (W0 - 1) * KS * C;
        }
        O[c + z0 * C] = op==L_AVGPOOL ? v / (KS * KS) : v;
    }
    g.sync();
}

__KERN__ void k_filter(
    DU *I, DU *F, DU *O,                   ///< input, filter, output tensors
    int H, int W                           ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = blockIdx.z, C = gridDim.z;
    const int z0 = c + (i0 + j0 * W) * C;
    auto g = cg::this_thread_block();
    
    if (i0 < H && j0 < W && c < C) {
        O[z0] = (F[z0] > DU0) ? I[z0] : DU0;
    }
    g.sync();
}

__GPU__ Tensor&
Model::onehot() {
    Tensor &out = (*this)[-1];                             ///< model output
    Tensor &hot = _mmu->tensor(out.numel).fill(DU0);       ///< one-hot vector
    if (!_dset) {
        ERROR("Model#loss dataset not set yet?\n");
        return hot;
    }
    int dsz = hot.H() * hot.W() * hot.C();                 ///< layer size
    DU *v   = &_dset->label[_dset->N() * _dset->batch_id]; ///< target labels
    DU *h   = hot.data;
    for (int n=0; n < out.N(); n++, h+=dsz) {              /// * setup one-hot
        U32 i = INT(v[n]);
        h[i < dsz ? i : 0] = DU1;
    }
    return hot;
}

__GPU__ Model&
Model::forward(Tensor &input) {
    Tensor &n1 = (*this)[1];        ///< reference model input layer
    if (!n1.is_same_shape(input)) {
        ERROR("Model#forward dataset dim != model input dim?\n");
        return *this;
    }
    if (input.is_dataset()) {       /// * if source is a dataset
        input.ref_inc();            /// * increase data
        _dset = (Dataset*)&input;   /// * set current dataset
        _hot  = &onehot();          /// * cache batch one-hot vectors
    }
    n1 = input;                     /// * copy into model first layer input
    ///
    /// cascade execution layer by layer forward
    /// TODO: model execution becomes a superscalar pipeline
    ///
    auto trace = [](int i, Tensor &in, Tensor &out) {
        printf("%2d> %s Σ=%6.2f [%d,%d,%d,%d]\tp=%-2d => out[%d,%d,%d,%d]",
            i, d_nname(in.grad_fn), in.sum(),
            in.N(), in.H(), in.W(), in.C(), in.parm,
            out.N(), out.H(), out.W(), out.C());
    };
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        trace(i, in, out);
        _fstep(in, out);
        printf("\n");
    }
    return *this;
}
/// ========================================================================
/// private methods 
///
__GPU__ void
Model::_fstep(Tensor &in, Tensor &out) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);              ///< default blocks
    dim3 grd(TGRID(out.W(), out.H(), out.C(), blk));  ///< default grids
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                 ///< layer function
    switch(fn) {
    case L_CONV:    _fconv(in, out);   break; ///< convolution
    case L_LINEAR:  _flinear(in, out); break; ///< out = W @ in + B
    case L_FLATTEN: out = in;          break; /// * straight copy
    case L_RELU: {
        const int N = out.N(), H0 = out.H(), W0 = out.W();
        for (int n = 0; n < N; n++) {
            DU *d1 = in.slice(n), *d0 = out.slice(n);
            k_filter<<<grd, blk>>>(d1, d1, d0, H0, W0);
            GPU_SYNC();
        }
    } break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: {                        /// * feed to CrossEtropy
        out = in;                            /// * copy content for exp calc
        /// TODO: mini-batch, in[100,10] => sum[100]
        out.map(O_EXP);
        for (int n = 0; n < in.N(); n++) {
            DU sum = out.data[n * in.HWC()];       /// * sum all log proba.
            out *= DU1/(sum + DU_EPS);           /// * divide by sum(exp)
        }
        printf(" Σ=%5.3f", out.sum());       /// * verify sum
    } break;
    case L_LOGSMAX: {                        /// * feed to NLL
        out = in;                            /// * use out as tmp
        DU sum = LOG(out.map(O_EXP).sum());  /// * calc logsum
        out = in;                            /// * overwrite out again
        out -= sum;                          /// * Xi - logsum
        printf(" Σ=%5.3f", out.sum());       /// * verify sum
    } break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: _fpool(in, out,fn); break;
    case L_DROPOUT:
        const int H0 = out.H(), W0 = out.W();
        for (int n = 0; n < out.N(); n++) {
            DU *m  = in.grad[0]->slice(n);   /// * mask data
            DU *d1 = in.slice(n), *d0 = out.slice(n);
            k_filter<<<grd, blk>>>(d1, m, d0, H0, W0);
        }
        GPU_SYNC();
        break;
    }
    debug(out);
}

#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14 */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12 */

__GPU__ int
Model::_fconv(Tensor &in, Tensor &out) {
    Tensor &tf = *in.grad[0];              ///< filter tensor
    Tensor &tb = *in.grad[1];              ///< bias tensor
    int C5 = tf.parm;                      ///< 5th dimension C1
    
    printf(" f[%d][%d,%d,%d,%d], b[%d]", C5, tf.N(), tf.H(), tf.W(), tf.C(), tb.numel);
        
    const int H0 = out.H(), W0 = out.W(), C0 = out.C();  ///< outpt dimensions
    const int C1 = in.C();
                    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);                 ///< default blocks
    dim3 g3((W0 + TILE3 - 1) / TILE3, (H0 + TILE3 - 1) / TILE3, C0);
    dim3 g5((W0 + TILE5 - 1) / TILE5, (H0 + TILE5 - 1) / TILE5, C0);

    int ks = tf.H();
    for (int n = 0; n < out.N(); n++) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        DU *f  = tf.slice(n), *b  = tb.data;
        switch(ks) {                       /// * TODO: handles rectangular filters
        case 3: k_conv2d<TILE3,3><<<g3,blk>>>(d1, f, b, d0, H0, W0, C1); break;
        case 5: k_conv2d<TILE5,5><<<g5,blk>>>(d1, f, b, d0, H0, W0, C1); break;
        default:
            ERROR("model_fwd#conv kernel_size=%d not supported\n", ks);
            return -1;
        }
    }
    GPU_SYNC();
    return 0;
}

__GPU__ int
Model::_flinear(Tensor &in, Tensor &out) {
    Tensor &tw = *in.grad[0];                         ///< weight tensor
    Tensor &tb = *in.grad[1];                         ///< bias tensor
    const int Hw = tw.H(), Ww = tw.W();               ///< dense layer dims
    
    printf(" w[%d,%d,%d,%d] @ in[%d,%d,%d,%d] + b[%d]",
        tw.N(), tw.H(), tw.W(), tw.C(),
        in.N(), in.H(), in.W(), in.C(), tb.numel);
        
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);             ///< default blocks
    dim3 grd(TGRID(out.W(), out.H(), out.C(), blk)); ///< default grids

    DU *w = tw.data, *b = tb.data;
    for (int n = 0; n < out.N(); n++, w += tw.HWC()) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        for (int y = 0; y < Hw; y++) {               /// TODO: kernel version
            d0[y] = b[y];                            /// init with bias
            for (int x = 0; x < Ww; x++) {           /// dot product
                d0[y] += w[x + y * Ww] * d1[x];
            }
        }
    }
    /*
      if (out.numel < 20) dump(d0, 1, out.numel, 1);
      else {
      int k = SQRT(out.numel);
      -dump(d0, k+1, k, 1);
      }
    */
    return 0;
}

__GPU__ int
Model::_fpool(Tensor &in, Tensor &out, t4_layer fn) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);                ///< default blocks
    dim3 grd(TGRID(out.W(), out.H(), out.C(), blk));    ///< default grid
    
    const int N  = out.N(), H0 = out.H(), W0 = out.W(); ///< output dimensions
    const int ks = in.parm;                             ///< kernel size
    
    for (int n = 0; n < N; n++) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        switch(ks) {                         /// pooling kernel size
        case 0x2: k_pool<2><<<grd,blk>>>(d1, d0, H0, W0, fn); break;
        case 0x3: k_pool<3><<<grd,blk>>>(d1, d0, H0, W0, fn); break;
        default:
            ERROR("model#pooling kernel_size=%d not supported\n", ks);
            return -1;
        }
    }
    GPU_SYNC();
    return 0;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
