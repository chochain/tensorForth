/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model Backward Propagation implementation
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
    __shared__ DU it[T4_WARP_SQ];                    ///< input cache [16x16]
    __shared__ DU ot[T4_WARP_SQ];                    ///< output cache [16x16]
    __shared__ DU df[TS * TS * KS * KS];             ///< df cache [12x12x3x3]
    
    const int tx = threadIdx.x, j1 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i1 = ty + blockIdx.y * TS;
    const int C1 = gridDim.z,   c1 = blockIdx.z;     ///< input channels
    const int z1 = c1 + (j1 + i1 * W) * C1;          ///< input array index
    const int zt = tx + ty * T4_WARP_SZ;             ///< tile index
    ///
    /// process z1, i.e. [TS, TS, C1] cells per kernel call
    ///
    const int i0 = i1 - int(KS / 2);                 ///< dY coordinates
    const int j0 = j1 - int(KS / 2);

    auto g = cg::this_thread_block();                ///< group all threads

    it[zt] = (i1 < H && j1 < W) ? I[z1] : DU0;       ///< cached input tile
    g.sync();
    
    for (int c0 = 0; c0 < C0; c0++) {                ///< each dY channel
        ot[zt] =                                     /// * cache dY tile
            (i0 >= 0 && i0 < H && j0 >= 0 && j0 < W) /// * with zero padding
            ? O[c0 + (j0 + i0 * W) * C0] : DU0;      /// * by channel
        g.sync();                                    /// * smem write barrier
        ///
        /// dX = sum(F * dY)
        /// dF = sum(dY * X)
        ///
        DU sum = DU0;
        const int zf = (c1 + c0 * C1) * KS * KS;     ///< filter index
        if (tx < TS && ty < TS) {                    /// * within tile [12x12]
            DU *fx = &F[zf + C1 * (KS * KS - 1)];    ///< F[KS-1,KS-1] rot180
            DU *dfx= &df[(tx + ty * TS) * KS * KS];  ///< df cache ptr
            for (int y = 0; y < KS; y++) {           /// * process one cell
                for (int x = 0; x < KS; x++, fx -= C1) {
                    int k = zt + x + y * T4_WARP_SZ;
                    sum      += (*fx) * ot[k];       /// * dX += F * dY
                    *(dfx++) =  ot[k] * it[k];       /// * df = dY * X
                }
            }
            if (i1 < H && j1 < W) {                  /// * update input matrix
                if (c0==0) I[z1] = sum;              /// * no bias
                else       I[z1] += sum;             /// * accumulate all c0
            }
        }
        g.sync();                                    /// * d read barrier
        ///
        /// collect dF (= dY * X), KS * KS threads
        ///
        if (tx < KS && ty < KS) {                    /// * TODO: CDP scan
            DU *DFx = &DF[c1 + (tx + (ty + c0 * KS) * KS) * C1];
            DU *dfx = &df[tx + ty * KS];
            for (int i = 0; i < TS * TS; i++, dfx += KS * KS) {
                *DFx += *dfx;                        /// dF += df (= dY * X)
            }
        }
        g.sync();                           /// * d read barrier
    }
}
template<int KS>                            /// kernel size
__KERN__ void k_dpool(
    DU *I, DU *O,                           ///< input, output buffers
    int H0, int W0,                         ///< output HW (C1==C0)
    t4_layer op
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = blockIdx.z, C = gridDim.z;
    const int z0 = j0 + i0 * W0;            ///< output matrix index
    const int z1 = (j0 + i0 * W0 * KS) * KS;///< input tensor index
    const int zc = c + z0 * C;              ///< output tensor index
    auto g = cg::this_thread_block();
    
    if (i0 < H0 && j0 < W0 && c < C) {
        DU *ix = &I[c + z1 * C], *t = ix;
        DU2 v  = (op != L_AVGPOOL) ? *ix : O[zc] / (KS * KS);
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
            ix += (W0 - 1) * KS * C;
        }
        if (op != L_AVGPOOL) *t = O[zc];   /// * update arg cell
    }
    g.sync();
}

__KERN__ void k_dfilter(
    DU *I, DU *F, DU *O,                   ///< input, filter, output
    int H, int W                           ///< H1==H0, W1==W0 (C1==C0)
    ) {
    const int j1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i1 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c  = blockIdx.z, C = gridDim.z;
    const int z1 = c + (i1 + j1 * W) * C;
    auto g = cg::this_thread_block();
    
    if (i1 < H && j1 < W && c < C) {
        I[z1] = (F[z1] > DU0) ? O[z1] : DU0;
    }
    g.sync();
}

__GPU__ Model&
Model::backprop(Tensor &tgt) {
    Tensor &nx = (*this)[numel - 1];
    if (!nx.is_same_shape(tgt)) {
        ERROR("Model#backprop target dim?\n");
        return *this;
    }
    ///
    /// cascade execution layer by layer backward
    ///
    Tensor::copy(tgt, nx);
    for (U16 i = numel - 2; i > 0; i--) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        printf("%2d> %s [%d,%d,%d]\tp=%2d <= out'Σ=%6.2f [%d,%d,%d] ",
            i, d_nname(in.grad_fn),
            in.H(), in.W(), in.C(), in.parm,
            out.sum(), out.H(), out.W(), out.C());
        _bstep(in, out);
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
Model::_bstep(Tensor &in, Tensor &out) {
    DU   *d1 = in.data, *d0 = out.data;              ///< input, output data
    int  H1 = in.H(), W1 = in.W(), C1 = in.C();      ///< input HWC
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(W1, H1, C1, blk));

    auto dump = [](DU *v, int H, int W, int C) {
        for (int k = 0; k < C; k++) {
            printf("\nC=%d ---\n", k);
            DU sum = DU0;
            for (int i = 0; i < H; i++) {
                DU isum = DU0;
                for (int j = 0; j < W; j++) {
                    DU x = v[k + (j + i * W) * C];
                    isum += x;
                    printf("%5.2f", x);
                }
                printf(" Σ=%6.3f\n", isum);
                sum += isum;
            }
            printf(" ΣΣ=%6.3f\n", sum);
        }
    };
    auto dump_dbdf = [C1](DU *df, DU *db, int C0, int fsz) {
        DU sum = DU0;
        printf("\n\tdb=");
        for (int c0 = 0; c0 < C0; c0++) {
            printf("%6.3f ", db[c0]);
            sum += db[c0];
        }
        printf("Σ=%6.3f", sum);
        for (int c1 = 0; c1 < C1; c1++) {
            printf("\n\tdf[%d]=", c1);
            sum = DU0;
            for (int i=0; i<fsz; i++, df++) {
                sum += *df;
                printf("%6.3f", *df);
            }
            printf(" Σ=%6.3f", sum);
        }
    };
    auto conv = [d1, d0, H1, W1, C1, blk](int C0, int ks, DU *f, DU *df, DU *db) {
        dim3 g3((W1 + TILE3 - 1) / TILE3, (H1 + TILE3 - 1) / TILE3, C1);
        dim3 g5((W1 + TILE5 - 1) / TILE5, (H1 + TILE5 - 1) / TILE5, C1);
        switch (ks) {
        case 3: k_dconv2d<TILE3,3><<<g3,blk>>>(d1, f, df, d0, H1, W1, C0); break;
        case 5: k_dconv2d<TILE5,5><<<g5,blk>>>(d1, f, df, d0, H1, W1, C0); break;
        default: return -1;
        }
        /// accumulate dB = sum(dY), TODO: CDP
        for (int c0 = 0; c0 < C0; c0++, db++) {
            DU *ox = d0 + c0;
            for (int k = 0; k < H1 * W1; k++, ox+=C0) *db += *ox;
        }
        return 0;
    };
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                 ///< layer function
    switch(fn) {
    case L_CONV:   {
        Tensor &f = *in.grad[0], &df = *in.grad[2]; ///< filter tensor
        Tensor &b = *in.grad[1], &db = *in.grad[3]; ///< bias tensor
        const int C1 = f.parm, Nf = f.N(), Hf = f.H(), Wf = f.W(), Cf = f.C();
        printf(" f[%d][%d,%d,%d,%d], b[%d]", C1, Nf, Hf, Wf, Cf, b.numel);
        if (conv(out.C(), Hf, f.data, df.data, db.data)) {
            ERROR("model_back#conv kernel_size %d not supported\n", Hf);
        }
        dump_dbdf(df.data, db.data, out.C(), Nf * Hf * Wf * Cf);
        printf("\nin[%d,%d,%d]=", H1, W1, C1); dump(d1, H1, W1, C1);
    } break;
    case L_LINEAR: {                          ///< out = w @ in + b
        Tensor &w  = *in.grad[0];             ///< weight tensor
        Tensor &dw = *in.grad[2];             ///< d_weight tensor
        Tensor &db = *in.grad[3];             ///< d_bias tensor
        int H0 = out.H(), M = w.H(), N = w.W();///< fully connected dimensions
        /// dw += out[10,1] @ in^t[1,49]
        /// in = w^t[49,10] @ out[10,1]
        printf("\n\tdw[%d,%d] += out'[%d,1] @ in^t[1,%d]", M, N, H0, H1);
        printf("\n\tin[%d, 1]  = w^t[%d,%d] @ out'[%d,1]", H1, N, M, H0);
        db += out;
        Tensor::mm(out, in, dw, (t4_mm_opt)(MM_INC | MM_B_TXP));
        Tensor::mm(w, out, in, MM_A_TXP);
    } break;
    case L_FLATTEN: Tensor::copy(out, in); break;  /// * pass dY to X
    case L_RELU:    k_dfilter<<<grd,blk>>>(d1, d1, d0, H1, W1); break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: in -= out; /* delta */ break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: {
        U16 ks = in.parm;                              ///< pooling kernel size
        U16 W0 = out.W(), H0 = out.H(), C0 = out.C();  ///< output dimensions
        dim3 g(TGRID(W0, H0, C0, blk));
        switch(ks) {                           
        case 0x2: k_dpool<2><<<g,blk>>>(d1, d0, H0, W0, fn); break;
        case 0x3: k_dpool<3><<<g,blk>>>(d1, d0, H0, W0, fn); break;
        }
        if (H1 < 10) dump(d1, W1, H1, C1);
    } break;
    case L_DROPOUT:
        Tensor &msk = *in.grad[0];             ///< dropout mask
        k_dfilter<<<grd,blk>>>(d1, msk.data, d0, H1, W1);
        break;
    }
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
