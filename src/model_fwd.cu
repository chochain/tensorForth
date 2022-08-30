/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model Feed Forward implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
///
/// Row convolution filter
///
template<int TS, int KS>         ///> tile size, kernel size
__KERN__ void k_conv2d(
    DU *I, DU *F, DU *B, DU *O,  ///> input A[HxW], F[KxK] kernel B[C] bias, output C[HxW]
    int H, int W, int C1         ///< output HW, input channel & offset
    ) {
    __shared__ DU d[T4_WARP_SZ * T4_WARP_SZ];        ///< shared memory [16x16]
    
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * TS;
    const int C  = blockDim.z,  c0 = threadIdx.z;    ///< output channels
    const int z0 = c0 + (j0 + i0 * W) * C;           ///< output array index
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const int i  = i0 - int(KS / 2);                 ///< input coordinates
    const int j  = j0 - int(KS / 2);

    for (int c1 = 0; c1 < C1; c1++) {                ///< each input channel
        d[tx + ty * T4_WARP_SZ] =                    /// * cache input data
            (i >= 0 && i < H && j >= 0 && j < W)     /// * with zero padding
            ? I[c1 + (j + i * W) * C1] : DU0;        /// * by channel
        __syncthreads();                             /// * smem write barrier
        ///
        /// sum of element-wise multiplication
        ///
        int f1 = c1 * KS * KS * C;                   /// * dense [C1,C] filter
        if (tx < TS && ty < TS) {                    /// * within tile [12x12]
            DU sum = DU0;
            for (int y = 0; y < KS; y++) {           /// * process one cell
                int d0 = tx + (y + ty) * T4_WARP_SZ; ///< offset to smem
                int b0 = (y * KS) + f1;              ///< offset to filter
                for (int x = 0; x < KS; x++) {
                    sum += F[x + b0] * d[x + d0];
                }
            }
            if (i0 < H && j0 < W) {                  /// * update output matrix
                if (c1==0) O[z0] = sum + B[c0];      /// * O[ijk] with bias
                else       O[z0] += sum;
            }
        }
        __syncthreads();                             /// * d read barrier
    }
}

template<int KS>                           /// kernel size
__KERN__ void k_pool(
    DU *I, DU *O,
    int H, int W, int C,                   /// HWC (C preserved)
    t4_layer op
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c0 = threadIdx.z + blockIdx.z * blockDim.z;
    const int z0 = j0 + i0 * W;            ///< output array index
    const int z1 = j0 + i0 * W * KS;       ///< input array index 
    
    if (i0 < H && j0 < W && c0 < C) {
        DU *d  = &I[c0 + z1 * KS * C];
        DU2 v  = op==L_AVGPOOL ? DU0 : *d;
        for (int y = 0; y < KS; y++) {
            for (int x = 0; x < KS; x++) {
                DU dx = *d;
                switch (op) {
                case L_MAXPOOL: v = MAX(dx, v); break;
                case L_AVGPOOL: v += dx;        break;
                case L_MINPOOL: v = MIN(dx, v); break;
                }
                d += C;                   
            }
            d += (W - 1) * KS * C;
        }
        O[c0 + z0 * C] = op==L_AVGPOOL ? v / (KS * KS) : v;
    }
}

__KERN__ void k_filter(
    DU *I, DU *F, DU *O,                   ///< input, filter, output tensors
    int H, int W, int C                    ///< HWC
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c0 = threadIdx.z + blockIdx.z * blockDim.z;
    const int z0 = c0 + (i0 + j0 * W) * C;
    
    if (i0 < H && j0 < W && c0 < C) {
        O[z0] = (F[z0] > DU0) ? I[z0] : DU0;
    }
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
        printf("%2d> %s [%d,%d,%d] p=%d =>",
               i, d_nname(in.grad_fn), in.H(), in.W(), in.C(), in.parm);
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
    int  H = out.H(), W = out.W(), C = out.C();      ///< HWC
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, C), grd(        ///< GPU warp size setup
        (W + blk.x - 1) / blk.x,
        (H + blk.y - 1) / blk.y
    );
    auto conv = [d1, d0, H, W, C, blk](U16 C1, U16 ks, DU *f, DU *b) {
        dim3 g3((W+TILE3-1)/TILE3, (H+TILE3-1)/TILE3);
        dim3 g5((W+TILE5-1)/TILE5, (H+TILE5-1)/TILE5);
        switch(ks) {            /// * TODO: handles rectangular filters
        case 3: k_conv2d<TILE3,3><<<g3,blk>>>(d1, f, b, d0, H, W, C1); break;
        case 5: k_conv2d<TILE5,5><<<g5,blk>>>(d1, f, b, d0, H, W, C1); break;
        default: return -1;
        }
        return 0;
    };
    auto linear = [d1, d0](int M, int N, DU *w, DU *b) {
        for (int y = 0; y < M; y++) {        /// TODO: kernel version
            int yn = y * N;
            d0[y] = b[y];                    /// init with bias
            for (int x = 0; x < N; x++) {    /// dot product
                d0[y] += w[x + yn] * d1[x];
            }
        }
    };
    auto pool = [d1, d0, H, W, C, blk, grd](int ks, t4_layer fn) {
        switch(ks) {           /// pooling kernel size
        case 0x2: k_pool<2><<<grd,blk>>>(d1, d0, H, W, C, fn); break;
        case 0x3: k_pool<3><<<grd,blk>>>(d1, d0, H, W, C, fn); break;
        default: return -1;
        }
        return 0;
    };
    auto dump = [](DU *v, int H, int W, int C) {
        for (int k = 0; k < C; k++) {
            printf("\nC=%d ---", k);
            for (int i = 0; i < H; i++) {
                printf("\n");
                for (int j = 0; j < W; j++) {
                    DU x = v[k + (j + i * W) * C];
                    printf(x < DU0 ? "%.2f" : " %.2f", x);
                }
            }
        }
        printf("\n");
    };
    ///
    /// layer function dispatcher
    ///
    printf(" out[%d,%d,%d]", H, W, C);
    t4_layer fn = in.grad_fn;                 ///< layer function
    switch(fn) {
    case L_CONV:   {
        Tensor &f = *in.grad[0];              ///< filter tensor
        Tensor &b = *in.grad[1];              ///< bias tensor
        printf(" f[%d][%d,%d,%d,%d], b[%d]",
               f.parm, f.N(), f.H(), f.W(), f.C(), b.numel);
        if (conv(in.C(), f.H(), f.data, b.data)) {
            ERROR("model#conv kernel_size=%d not supported\n", f.H());
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
    case L_RELU:    k_filter<<<grd, blk>>>(d1, d1, d0, H, W, C); break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: {
        Tensor &t = *in.grad[0];             ///< tmp tensor
        Tensor::copy(in, t);                 /// * copy content for exp calc
        DU sum = t.map(O_EXP).sum() + DU_EPS;/// * sum all probabilities
        Tensor::mat(O_MUL, t, DU1/sum, out); /// * p / sum(p)
        printf(" sum=%.3f", out.sum());      /// * verify sum
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
        k_filter<<<grd, blk>>>(d1, msk.data, d0, H, W, C);
        break;
    }
    cudaDeviceSynchronize();
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
