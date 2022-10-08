/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

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
    const int c0 = threadIdx.z, C0 = blockDim.z;     ///< channel deep
    const int n  = blockIdx.z;                       ///< batch slice id
    const int ns0= n * H * W * C0;                   ///< output slice index
    const int ns1= n * H * W * C1;                   ///< input slice index
    const int z0 = c0 + (j0 + i0 * W) * C0 + ns0;    ///< output array index
    const int t0 = tx + ty * T4_WARP_SZ;             ///< tile index
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const int i1 = i0 - int(KS / 2);                 ///< input coordinates
    const int j1 = j0 - int(KS / 2);

    auto g = cg::this_thread_block();                ///< all threads of block
    for (int c1 = 0; c1 < C1; c1++) {                ///< each input channel
        const int z1 = c1 + (j1 + i1 * W) * C1 + ns1;
        it[t0] =                                     /// * cache input data
            (i1 >= 0 && i1 < H && j1 >= 0 && j1 < W) /// * with zero padding
            ? I[z1] : DU0;                           /// * by channel
        g.sync();                                    /// * smem write barrier
        ///
        /// Y = sum(W * X)
        /// TODO: cache F
        ///
        const int zf = c0 + c1 * C0 * KS * KS;       ///< filter index [C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                    /// * each tile
            DU sum = DU0;
            DU *fx = &F[zf], *ix = &it[t0];          /// * filter[0], tile[tx,ty]
            for (int y = 0; y < KS; y++) {           /// * each filter
                for (int x = 0; x < KS; x++) {
                    sum += (*fx) * ix[x];            /// Y += W * X
                    fx  += C0;                       /// * next filter cell
                }
                ix += T4_WARP_SZ;                    /// next row of tile
            }
            if (i0 < W && j0 < H) {
                if (c1==0) O[z0] = sum + B[c0];      /// * O[ijc] with bias
                else       O[z0] += sum;
            }
        }
        g.sync();
    }
}

template<int KS>                           /// kernel size
__KERN__ void k_pool(
    DU *I, DU *O,                          ///< input, output buffers
    int H, int W,                          ///< output HW (C0==C1)
    t4_layer op                            ///< pooling ops
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = threadIdx.z, C = blockDim.z;      ///< channel deep
    const int ns = blockIdx.z * H * W * C;           ///< batch slice idx
    const int z0 = c + (j0 + i0 * W) * C + ns;       ///< output array index
    const int z1 = c + ((j0 + (i0 * W) * KS) * KS) * C + ns * KS * KS;
    
    if (i0 < H && j0 < W && c < C) {
        DU *ix = &I[z1];
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
            ix += (W - 1) * KS * C;
        }
        O[z0] = op==L_AVGPOOL ? v / (KS * KS) : v;
    }
}

__KERN__ void k_filter(
    DU *I, DU *F, DU *O,                   ///< input, filter, output tensors
    int H, int W                           ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = threadIdx.z, C = blockDim.z;      ///< channel deep
    const int ns = blockIdx.z * H * W * C;           ///< batch slice idx
    const int z0 = c + (i0 + j0 * W) * C + ns;       ///< output tensor index
    
    if (i0 < H && j0 < W && c < C) {
        O[z0] = (F[z0] > DU0) ? I[z0] : DU0;
    }
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
            i, d_nname(in.grad_fn), in.sum() / in.N() / in.C(),
            in.N(), in.H(), in.W(), in.C(), in.parm,
            out.N(), out.H(), out.W(), out.C());
    };
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        trace(i, in, out);
        _fstep(in, out);
//        debug(out);
        printf("\n");
    }
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
    t4_layer fn = in.grad_fn;                     ///< layer function
    switch(fn) {
    case L_CONV:    _fconv(in, out);       break; ///< convolution
    case L_LINEAR:  _flinear(in, out);     break; ///< out = W @ in + B
    case L_FLATTEN: out = in;              break; ///< straight copy
    case L_RELU:    _ffilter(in, in, out); break; ///< filter in < 0 
    case L_TANH:    /* TODO */             break;
    case L_SIGMOID: /* TODO */             break;
    case L_SOFTMAX: _fsoftmax(in, out);    break; /// * feed to CrossEtropy
    case L_LOGSMAX: _flogsoftmax(in, out); break; /// * feed to NLL
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: _fpool(in, out, fn);   break;
    case L_DROPOUT: {                             ///< dropout mask 
        Tensor &msk = *in.grad[0];
        _ffilter(in, msk, out);
    } break;
    default: ERROR("Model#forward layer=%d not supported\n", fn);
    }
}

#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14 */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12 */

__GPU__ int
Model::_fconv(Tensor &in, Tensor &out) {
    Tensor &tf = *in.grad[0];                             ///< filter tensor
    Tensor &tb = *in.grad[1];                             ///< bias tensor
    
    printf(" f[%d,%d,%d,%d], b[%d]", tf.N(), tf.H(), tf.W(), tf.C(), tb.numel);
        
    const int N = out.N(), H = out.H(), W = out.W();      ///< outpt dimensions
    const int C0 = out.C(), C1 = in.C();                  ///< output, input channel deep
                    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, C0);                 ///< default blocks
    dim3 g3((W + TILE3 - 1) / TILE3, (H + TILE3 - 1) / TILE3, N);
    dim3 g5((W + TILE5 - 1) / TILE5, (H + TILE5 - 1) / TILE5, N);

    DU *d1 = in.data, *d0 = out.data, *f = tf.data, *b = tb.data;
    int ks = tf.H();
    switch(ks) {                       /// * TODO: handles rectangular filters
    case 3: k_conv2d<TILE3,3><<<g3,blk>>>(d1, f, b, d0, H, W, C1); break;
    case 5: k_conv2d<TILE5,5><<<g5,blk>>>(d1, f, b, d0, H, W, C1); break;
    default:
        ERROR("model_fwd#conv kernel_size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();
    return 0;
}

__GPU__ int
Model::_flinear(Tensor &in, Tensor &out) {
    Tensor &tw = *in.grad[0];                         ///< weight tensor
    Tensor &tb = *in.grad[1];                         ///< bias tensor

    const int N  = out.N();                           ///< batch size (N1 == N0)
    const int C0 = tw.H(), C1 = tw.W();               ///< dense layer dims
    
    printf(" = w[%d,%d] @ in[%d,%d,%d,%d] + b[%d]",
        C0, C1, in.N(), in.H(), in.W(), in.C(), tb.numel);
        
    DU *w = tw.data, *b = tb.data;
    for (int n = 0; n < N; n++) {                     /// * walk through batch
        DU *x = in.slice(n), *y = out.slice(n);
        for (int i = 0; i < C0; i++) {                /// TODO: kernel version
            y[i] = b[i];                              /// init with bias
            for (int j = 0; j < C1; j++) {            /// dot product
                y[i] += w[j + i * C1] * x[j];         /// * Y += W @ X + B
            }
        }
    }
    return 0;
}

__GPU__ int
Model::_ffilter(Tensor &in, Tensor &msk, Tensor &out) {
    const int N = out.N(), H = out.H(), W = out.W();
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, out.C());          ///< default blocks
    dim3 grd(NGRID(W, H, out.N(), blk));                ///< default grids
        
    k_filter<<<grd, blk>>>(in.data, msk.data, out.data, H, W);
    GPU_SYNC();
    
    return 0;
}

__GPU__ int
Model::_fpool(Tensor &in, Tensor &out, t4_layer fn) {
    const int H = out.H(), W = out.W();                 ///< output dimensions
    const int ks = in.parm;                             ///< kernel size
    
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, out.C());          ///< default blocks
    dim3 grd(NGRID(W, H, out.N(), blk));                ///< default grid
    
    switch(ks) {                                        /// pooling kernel size
    case 0x2: k_pool<2><<<grd,blk>>>(in.data, out.data, H, W, fn); break;
    case 0x3: k_pool<3><<<grd,blk>>>(in.data, out.data, H, W, fn); break;
    default:
        ERROR("model#pooling kernel_size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();
    
    return 0;
}

__GPU__ int
Model::_fsoftmax(Tensor &in, Tensor &out) {   /// * TODO: DCP
    out = in;                                 /// * copy content for exp calc
    DU *sum = in.grad[0]->data;               /// * sum(exp(xi)), for DCP
    int sz  = in.HWC();
    for (int n = 0; n < in.N(); n++, sum++) { /// * loop throught batch
        *sum = DU0;
        DU *d = out.slice(n), *d1 = d;
        for (int i = 0; i < sz; i++) {
            *d   = EXP(*d);                   /// * softmax = exp(xi) / sum(exp(xi))
            *sum += *d++;
        }
        DU r = DU1 / (*sum + DU_EPS);         /// * r = 1.0 / sum(exp(xi))
        *sum = DU0;
        for (int i = 0; i < sz; i++) {
            *d1  *= r;
            *sum += *d1++;
        }
        printf(" Σ%d=%5.3f", n, *sum);        /// * verify sum
    }
    return 0;
}

__GPU__ int
Model::_flogsoftmax(Tensor &in, Tensor &out) {/// * TODO: DCP
    out = in;                                 /// * copy in data to out
    DU *sum = in.grad[0]->data;               /// * sum(exp(xi)), for DCP
    int sz  = in.HWC();
    for (int n = 0; n < in.N(); n++, sum++) { /// * loop throught batch
        *sum = DU0;
        DU *d = out.slice(n), *d1 = d;
        for (int i = 0; i < sz; i++) {
            *sum += EXP(*d++);
        }
        DU logsum = LOG(*sum);
        for (int i = 0; i < sz; i++) {
            *d1++ -= logsum;
        }
        printf(" lnΣ%d=%5.3f", n, logsum);   /// * verify sum
    }
    return 0;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
