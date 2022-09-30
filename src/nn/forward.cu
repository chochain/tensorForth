/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model Feed Forward implementation
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

__GPU__ Model&
Model::forward(Tensor &input) {
    Tensor &n1 = (*this)[1];
    if (!n1.is_same_shape(input)) {
        ERROR("Model#forward input dim?\n");
        return *this;
    }
    Tensor::copy(input, n1);       /// * feed input into model
    ///
    /// cascade execution layer by layer forward
    /// TODO: model execution becomes a superscalar pipeline
    ///
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        printf("%2d> %s Σ=%6.2f [%d,%d,%d]\tp=%d => out[%d,%d,%d]",
            i, d_nname(in.grad_fn), in.sum(),
            in.H(), in.W(), in.C(), in.parm,
            out.H(), out.W(), out.C());
        _fstep(in, out);
        printf("\n");
    }
    return *this;
}
/// ========================================================================
/// private methods 
///
#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14 */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12 */

__GPU__ void
Model::_fstep(Tensor &in, Tensor &out) {
    DU   *d1 = in.data, *d0 = out.data;              ///< input, output data
    int  H0 = out.H(), W0 = out.W(), C0 = out.C();   ///< output HWC
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(W0, H0, C0, blk));

    auto conv = [d1, d0, H0, W0, C0, blk](U16 C1, U16 ks, DU *f, DU *b) {
        dim3 g3((W0 + TILE3 - 1) / TILE3, (H0 + TILE3 - 1) / TILE3, C0);
        dim3 g5((W0 + TILE5 - 1) / TILE5, (H0 + TILE5 - 1) / TILE5, C0);
        switch(ks) {            /// * TODO: handles rectangular filters
        case 3: k_conv2d<TILE3,3><<<g3,blk>>>(d1, f, b, d0, H0, W0, C1); break;
        case 5: k_conv2d<TILE5,5><<<g5,blk>>>(d1, f, b, d0, H0, W0, C1); break;
        default: return -1;
        }
        return 0;
    };
    auto linear = [d1, d0](int M, int N, DU *w, DU *b) {
        for (int y = 0; y < M; y++) {        /// TODO: kernel version
            d0[y] = b[y];                    /// init with bias
            for (int x = 0; x < N; x++) {    /// dot product
                d0[y] += w[x + y * N] * d1[x];
            }
        }
    };
    auto pool = [d1, d0, H0, W0, blk, grd](int ks, t4_layer fn) {
        /// Note: H, W are output dimensions
        switch(ks) {                         /// pooling kernel size
        case 0x2: k_pool<2><<<grd,blk>>>(d1, d0, H0, W0, fn); break;
        case 0x3: k_pool<3><<<grd,blk>>>(d1, d0, H0, W0, fn); break;
        default: return -1;
        }
        return 0;
    };
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                 ///< layer function
    switch(fn) {
    case L_CONV:   {
        Tensor &f = *in.grad[0];              ///< filter tensor
        Tensor &b = *in.grad[1];              ///< bias tensor
        U16 Nf = f.N(), Hf = f.H(), Wf = f.W(), Cf = f.C();
        printf(" f[%d][%d,%d,%d,%d], b[%d]", f.parm, Nf, Hf, Wf, Cf, b.numel);
        
        if (conv(in.C(), Hf, f.data, b.data)) {
            ERROR("model_fwd#conv kernel_size=%d not supported\n", Hf);
        }
    } break;
    case L_LINEAR: {                          ///< out = w @ in + b
        Tensor &w = *in.grad[0];              ///< weight tensor
        Tensor &b = *in.grad[1];              ///< bias tensor
        int M = w.H(), N = w.W();             ///< fully connected dimensions
        printf(" w[%d,%d] @ in[%d] + b[%d]", M, N, in.numel, b.numel);
        linear(M, N, w.data, b.data);         ///< out = W @ in + B

        if (out.numel < 20) dump(d0, 1, out.numel, 1);
        /*
        else {
            int k = SQRT(out.numel);
            dump(d0, k+1, k, 1);
        }
        */
    } break;
    case L_FLATTEN: Tensor::copy(in, out); break;
    case L_RELU:    k_filter<<<grd, blk>>>(d1, d1, d0, H0, W0); break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: {
        Tensor &t = *in.grad[0];             ///< tmp tensor
        Tensor::copy(in, t);                 /// * copy content for exp calc
        DU sum = t.map(O_EXP).sum() + DU_EPS;/// * sum all probabilities
        Tensor::matx(O_MUL, t, DU1/sum, out);/// * p / sum(p)
        printf(" Σ=%5.3f", out.sum());       /// * verify sum
    } break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: {
        U16 ks = in.parm;                    ///< kerneal_size
        if (pool(ks, fn)) {
            ERROR("model#pooling kernel_size=%d not supported\n", ks);
        }
    } break;
    case L_DROPOUT:
        Tensor &msk = *in.grad[0];
        k_filter<<<grd, blk>>>(d1, msk.data, d0, H0, W0);
        break;
    }
    if (W0 > 1) view(out.data, H0, W0, C0);
    else        dump(out.data, W0, H0, C0);
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
