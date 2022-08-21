/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
///
/// Row convolution filter
///
template<int TS, int KS, int CS> ///> tile size, kernel size, 1:grey, 3:RGB
__KERN__ void k_conv(
    DU *I, DU *F, DU *B, DU *O,  ///> input A[HxW], F[KxK] kernel B[C] bias, output C[HxW]
    int H, int W                 ///< HWC
    ) {
    __shared__ DU d[CS][T4_WARP_SZ * T4_WARP_SZ];  ///< shared memory [3][16x16]
    
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * TS;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * TS;
    const int tz = threadIdx.z, c0 = tz + blockIdx.z * blockDim.z;
    const int i  = i0 - int(KS / 2);         ///< transfrom to input coordinate
    const int j  = j0 - int(KS / 2);
    const int z0 = c0 + (j0 + i0 * W) * CS;  ///< output array index

    for (int c = 0; c < CS; c++) {           /// channel depth
        d[c][tx + ty * T4_WARP_SZ] =         ///< shared memory with zero padding
            (i >= 0 && i < H && j >= 0 && j < W)
            ? I[c + (j + i * W) * CS] : DU0;
    }
    __syncthreads();
    ///
    /// sum of element-wise multiplication
    ///
    if (tx < TS && ty < TS) {                /// * within tile [12x12]
        DU sum = DU0;
        for (int c = 0; c < CS; c++) {               /// 3D filter
            for (int y = 0; y < KS; y++) {           /// * process one cell
                int d0 = tx + (y + ty) * T4_WARP_SZ; ///< offset to smem block
                int b0 = y * KS;
                for (int x = 0; x < KS; x++) {
                    sum += F[x + b0] * d[c][x + d0]; 
                }
            }
        }
        if (i0 < H && j0 < W && c0 < CS) {  /// * update C[i][j]
            O[z0] = sum + B[c0];            /// * update output matrix with bias
        }
    }
}

template<int KS>                           /// kernel size
__KERN__ void k_pooling(
    DU *I, DU *O,
    int H, int W, int C,                   /// HWC
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

__KERN__ void k_relu(
    DU *I, DU *O,
    int H, int W, int C                    ///< HWC
    ) {
    const int j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int c0 = threadIdx.z + blockIdx.z * blockDim.z;
    const int z0 = c0 + (i0 + j0 * W) * C;
    
    if (i0 < H && j0 < W && c0 < C) {
        O[z0] = (I[z0] >= DU0) ? I[z0] : DU0;
    }
}

__HOST__ const char*
Model::nname(int i) {               ///< network layer name
    static const char *name[] = {   /// double check with t4_layer
    "output ", "conv2d ", "linear ", "flatten", "relu   ",
    "tanh   ", "sigmoid", "softmax", "maxpool", "avgpool",
    "minpool", "dropout"
    };
    return name[i];
}
///
/// NN layer factory
///
__GPU__ Model&
Model::add(t4_layer fn, U16 n, DU bias, U16 *opt) {
    Tensor &in = (Tensor&)_mmu->du2obj(data[numel - 1]);
    if (!autograd || in.grad_fn != L_NONE) return *this;

    switch(fn) {
    case L_CONV:    _iconv(in, n, bias, opt);   break;
    case L_LINEAR:  _ilinear(in, n, bias);      break;
    case L_FLATTEN: _iflatten(in);              break;
    case L_RELU:
    case L_TANH:
    case L_SIGMOID: _icopy(in);                 break;
    case L_SOFTMAX: _isoftmax(in);              break;
    case L_MAXPOOL:
    case L_AVGPOOL:
    case L_MINPOOL: _ipooling(in, n);           break;
    case L_DROPOUT: _idropout(in, n);           break;
    }
    in.grad_fn = fn;
    return *this;
}

__GPU__ Model&
Model::forward(Tensor &input) {
    static const char *name[] = {   /// double check with t4_layer
    "output ", "conv2d ", "linear ", "flatten", "relu   ",
    "tanh   ", "sigmoid", "softmax", "maxpool", "avgpool",
    "minpool", "dropout"
    };
    Tensor &in = (*this)[1];
    if (!in.is_same_shape(input)) {
        ERROR("Model#forward input dim?\n");
        return *this;
    }
    Tensor::copy(input, in);       /// * feed input into model
    ///
    /// cascade execution layer by layer
    /// TODO: model execution becomes a superscalar pipeline
    ///
    for (int i = 2; i < numel; i++) {
        printf("%2d> %s [%d,%d,%d] p=%d =>",
               i-1, name[in.grad_fn], in.H(), in.W(), in.C(), in.parm); 
        Tensor &out = (*this)[i];
        _fstep(in, out);
        in = out;
        printf("\n");
    }
    return *this;
}
__GPU__ Model&
Model::backprop(Tensor &output) {
    return *this;
}
/// ========================================================================
/// private methods 
///
#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14 */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12 */

__GPU__ void
Model::_fstep(Tensor &in, Tensor &out) {
    DU   *da = in.data, *dc = out.data;              ///< input, output data
    int  H = out.H(), W = out.W(), C = out.C();      ///< HWC
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, C), grd(        ///< GPU warp size setup
        (W + blk.x - 1) / blk.x,
        (H + blk.y - 1) / blk.y
    );
    auto conv = [da, dc, H, W, C, blk](U16 kc, DU *f, DU *b) {
        dim3 g3((W+TILE3-1)/TILE3, (H+TILE3-1)/TILE3);
        dim3 g5((W+TILE5-1)/TILE5, (H+TILE5-1)/TILE5);
        switch(kc) {            /// * TODO: handles rectangular filters
        case 0x31: k_conv<TILE3,3,1><<<g3,blk>>>(da, f, b, dc, H, W); break;
        case 0x33: k_conv<TILE3,3,3><<<g3,blk>>>(da, f, b, dc, H, W); break;
        case 0x51: k_conv<TILE5,5,1><<<g5,blk>>>(da, f, b, dc, H, W); break;
        case 0x53: k_conv<TILE5,5,3><<<g5,blk>>>(da, f, b, dc, H, W); break;
        default: return -1;
        }
        return 0;
    };
    auto pooling = [da, dc, H, W, C, blk, grd](int ks, t4_layer fn) {
        switch(ks) {           /// pooling kernel size
        case 0x2: k_pooling<2><<<grd,blk>>>(da, dc, H, W, C, fn); break;
        case 0x3: k_pooling<3><<<grd,blk>>>(da, dc, H, W, C, fn); break;
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
                    printf("%.2f ", v[k + (j + i * W) * C]);
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
        U16 kc = f.H() << 4 | C;              ///< (kerneal_size, channel_depth)
        printf(" f[%d,%d,%d], b[%d]", f.H(), f.W(), f.C(), b.C());
        if (conv(kc, f.data, b.data)) {
            ERROR("model#conv kernel_size=0x%02x not supported\n", kc);
        }
        dump(dc, H, W, C);
    } break;
    case L_LINEAR: {                          ///< out = w @ in + b
        Tensor &w = *in.grad[0];  
        Tensor &b = *in.grad[1];
        printf(" w[%d,%d] @ in[%d,%d] + b[%d,%d]",
               w.H(), w.W(), in.H(), in.W(), b.H(), b.W());
        Tensor::copy(b, out);                 ///< add bias first
        Tensor::gemm(w, in, out, 1.0, 1.0);   ///< out += W * in
        dump(dc, (out.numel+6)/7, 7, 1);
    } break;
    case L_FLATTEN: Tensor::copy(in, out);                 break;
    case L_RELU:    k_relu<<<grd, blk>>>(da, dc, H, W, C); break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: {
        Tensor &t = *in.grad[0];             ///< tmp tensor
        Tensor::copy(in, t);                 /// * copy content for exp calc
        DU sum = t.map(O_EXP).sum() + DU_EPS;/// * sum all probabilities
        printf(" sum=%.2f ", sum);
        Tensor::mat(O_MUL, t, DU1/sum, out); /// * p / sum(p)
        dump(dc, 1, out.numel, 1);
    } break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: {
        U16 ks = in.parm;                    ///< kerneal_size
        if (pooling(ks, fn)) {
            ERROR("model#pooling kernel_size=%d not supported\n", ks);
        }
        dump(dc, H, W, C);
    } break;
    case L_DROPOUT: Tensor::copy(in, out); break;
    }
    cudaDeviceSynchronize();
}
__GPU__ void
Model::_bstep(Tensor &in, Tensor &out) {}
///
/// Convolution and Linear ops
///
__GPU__ void
Model::_iconv(Tensor &in, U16 C, DU bias, U16 *opt) {
    U16 M = opt[0], N = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : int((M-1)/2);      ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation
    U16 h = (in.H() - M + p*2) / s + 1;          ///> output height
    U16 w = (in.W() - N + p*2) / s + 1;          ///> output width
    if (M != N || (M != 3 && M != 5)) {
        ERROR("Model#conv2d f=[%d,%d]? 3x3 and 5x5 supported only.\n", M, N);
        return;
    }
    in.stride[0] = in.stride[1] = s;
    Tensor *f  = in.grad[0] = &tensor(1, M, N, C).map(O_FILL, DU1); ///> f
    Tensor *df = in.grad[2] = &tensor(1, M, N, C).map(O_FILL, DU0); ///> df
    Tensor *b  = in.grad[1] = &tensor(1, 1, 1, C).map(O_FILL, DU0); //bias); ///> b
    Tensor *db = in.grad[3] = &tensor(1, 1, 1, C).map(O_FILL, DU0);  ///> db
//    _mmu->random(*f, UNIFORM);                   /// * randomize f
//    Tensor::mat(O_SUB, *f, 0.5, *f);
    
    Tensor &out= tensor(1, h, w, C).map(O_FILL, DU0);  ///> output tensor
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_ilinear(Tensor &in, U16 n, DU bias) {
    U16 m = in.H();
    Tensor *w  = in.grad[0] = &tensor(1, n, m, 1).identity();  ///> w
    Tensor *dw = in.grad[2] = &tensor(1, n, m, 1).map(O_FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &vector(n).map(O_FILL, DU0); //bias);          ///> b
    Tensor *db = in.grad[3] = &vector(n).map(O_FILL, DU0);           ///> db
//    Tensor::mat(O_MUL, *w, 0.001, *w);
//    _mmu->random(*w, UNIFORM);                   /// * randomize w
    
    Tensor &out = vector(n);                     ///> output tensor sizing
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_iflatten(Tensor &in) {
    Tensor &out = vector(in.numel);
    in.parm     = in.numel;
    npush(out);
}
///
/// Activation ops
///
__GPU__ void
Model::_icopy(Tensor &in) {
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    npush(out);                   /// * stage for next stage
}
__GPU__ void
Model::_isoftmax(Tensor &in) {
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    in.grad[0] = &_mmu->copy(in); ///> tmp for exponental 
    npush(out);                   /// * stage for next stage
}
///
/// Pooling and Dropout ops
///
__GPU__ void
Model::_ipooling(Tensor &in, U16 f) {
    if (f != 2 && f != 3) {
        ERROR("Model#pooling f=[%d,%d]? 2x2 and 3x3 supported only\n", f, f);
        return;
    }
    in.parm = f;                  /// * keep pooling width
    U16 m = int((in.H() - f) / f) + 1;
    U16 n = int((in.W() - f) / f) + 1;
    U16 s[4] = { f, f, 1, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
    Tensor &out = tensor(1, m, n, in.C());
    npush(out);                 /// * stage for next stage
}
__GPU__ void
Model::_idropout(Tensor &in, U16 f) {
    Tensor &out = _mmu->copy(in);
    in.parm = f;
    npush(out);
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
