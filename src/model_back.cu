/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model Backward Propagation implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
///
/// Row convolution filter
///
template<int TS, int KS>         ///> tile size, kernel size
__KERN__ void k_dconv2d(
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
__KERN__ void k_dpooling(
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
                case L_MAXPOOL: if (dx > v) v = dx; break;
                case L_AVGPOOL: v += dx;            break;
                case L_MINPOOL: if (dx < v) v = dx; break;
                }
                d += C;                   
            }
            d += (W - 1) * KS * C;
        }
        O[c0 + z0 * C] = op==L_AVGPOOL ? v / (KS * KS) : v;
    }
}

__KERN__ void k_drelu(
    DU *I, DU *F, DU *O,                   ///< input, filter, output
    int H, int W, int C                    ///< HWC
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c0 = threadIdx.z + blockIdx.z * blockDim.z;
    const int z0 = c0 + (i0 + j0 * W) * C;
    
    if (i0 < H && j0 < W && c0 < C) {
        if (F[z0] > DU0) I[z0] = O[z0];
    }
}

__GPU__ Model&
Model::backprop(Tensor &tgt) {
    printf("here\n");
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
        printf("%2d> %s [%d,%d,%d] p=%d <=",
               i, d_nname(in.grad_fn), in.H(), in.W(), in.C(), in.parm); 
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
    int  H = in.H(), W = in.W(), C = in.C();         ///< input HWC
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, C), grd(        ///< GPU warp size setup
        (out.W() + blk.x - 1) / blk.x,
        (out.H() + blk.y - 1) / blk.y
    );
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
    printf(" out'[%d,%d,%d]", out.H(), out.W(), out.C());
    t4_layer fn = in.grad_fn;                 ///< layer function
    switch(fn) {
    case L_CONV:   {
        Tensor &f = *in.grad[0];              ///< filter tensor
        Tensor &b = *in.grad[1];              ///< bias tensor
        printf(" f[%d][%d,%d,%d,%d], b[%d]",
               f.parm, f.N(), f.H(), f.W(), f.C(), b.numel);
    } break;
    case L_LINEAR: {                          ///< out = w @ in + b
        Tensor &w  = *in.grad[0];             ///< weight tensor
        Tensor &dw = *in.grad[2];             ///< d_weight tensor
        Tensor &db = *in.grad[3];             ///< d_bias tensor
        int O = out.H(), M = w.H(), N = w.W();///< fully connected dimensions
        
        db += out;
        // dw += out[10,1] @ in^t[1,49]
        printf("\n\tdw[%d,%d] += out'[%d,1] @ in^t[1,%d] ", M, N, O, H);
        Tensor::mm(out, in, dw, (t4_mm_opt)(MM_INC | MM_B_TXP));
        // in = w^t[49,10] @ out[10,1]
        printf("\tin[%d,1] = w^t[%d,%d] @ out'[%d,1] ", H, N, M, O);
        Tensor::mm(w, out, in, MM_A_TXP);
    } break;
    case L_FLATTEN: Tensor::copy(out, in); break;
    case L_RELU:
        k_drelu<<<grd,blk>>>(d1, d1, d0, H, W, C);
        dump(d1, W, H, C);
        break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: in -= out; break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: {
        U16 ks2 = in.parm * in.parm;
        for (int j=0; j < out.numel; j++) {
            DU v = *d0++ / ks2;
            for (int i=0; i < ks2; i++) *d1++ += v;
        }
    } break;
    case L_DROPOUT:
        Tensor &msk = *in.grad[0];
        k_drelu<<<grd,blk>>>(d1, msk.data, d0, H, W, C);
    }
    cudaDeviceSynchronize();
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
