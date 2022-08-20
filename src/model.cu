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
template<int T, int K>           ///> TILE_SIZE, KERNEL_SIZE
__KERN__ void k_conv2d(          ///< TODO: C
    DU *A, DU *F, DU *B, DU *C,  ///> input A[MxN], F[KxK] kernel B[KxK] bias, output C[MxN]
    int M, int N
    ) {
    __shared__ DU d[T4_WARP_SZ * T4_WARP_SZ];  ///< shared memory [16x16]
    
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * T;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * T;
    const int z0 = j0 + i0 * N;              /// * output array index
    const int i  = i0 - int(K / 2);          ///< transfrom to input coordinate
    const int j  = j0 - int(K / 2);
    
    d[tx + ty * T4_WARP_SZ] =                ///< shared memory with zero padding
        (i >= 0 && i < M && j >= 0 && j < N)
        ? A[j + i * N] : DU0;
    __syncthreads();

    if (tx < T && ty < T) {                  /// * within tile [12x12]
        DU sum = DU0;
        #pragma unroll                       /// unroll to 3x3 or 5x5
        for (int y = 0; y < K; y++) {        /// * process one cell
            int d0 = tx + (y + ty) * T4_WARP_SZ; ///< offset to smem block
            int b0 = y * K;
            #pragma unroll
            for (int x = 0; x < K; x++) {
                sum += F[x + b0] * d[x + d0]; 
            }
        }
        if (i0 < M && j0 < N) {             /// * update C[i][j]
            C[z0] = sum + B[z0];            /// * update output matrix with bias
        }
    }
}

template<int K>
__KERN__ void k_pooling(                    ///< TODO: C
    DU *A, DU *C,
    int M, int N,
    t4_layer op
    ) {
    const int  j0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int  i0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int  z0 = j0 + i0 * N;            ///< output array index
    const int  z1 = (j0 + i0 * N * K) * K;  ///< input array index 
    
    if (i0 < M && j0 < N) {                 ///< TODO: CDP
        DU2 v = op==L_AVGPOOL ? DU0 : A[z1];
        DU *d = &A[z1];
        #pragma unroll
        for (int y = 0; y < K; y++, d += (N-1)*K) {
            #pragma unroll
            for (int x = 0; x < K; x++) {
                DU dx = *d++;
                switch (op) {
                case L_MAXPOOL: if (dx > v) v = dx; break;
                case L_AVGPOOL: v += dx;            break;
                case L_MINPOOL: if (dx < v) v = dx; break;
                }
            }
        }
        C[z0] = op==L_AVGPOOL ? v / (K*K) : v;
    }
}
        
__KERN__ void k_relu(                                        ///< TODO: C
    DU *A, DU *C, int M, int N) {
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int k = i + j * N;
    
    if (i < M && j < N) {                                    ///< TODO: CDP
        C[k] = (A[k] >= DU0) ? A[k] : DU0;
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
    case L_CONV2D:  _iconv2d(in, n, bias, opt); break;
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
        printf("%2d> %s [%d,%d] p=%d =>",
               i-1, name[in.grad_fn], in.H(), in.W(), in.parm); 
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
    int  m = out.H(), n = out.W(), p = in.parm;      ///< output sizing
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ), grd(           ///< GPU warp size
        (n + blk.x - 1) / blk.x,
        (m + blk.y - 1) / blk.y
    );
    auto dump = [](DU *v, int m, int n) {
        for (int i=0; i<m; i++) {
            printf("\n");
            for (int j=0; j<n; j++) printf("%.2f ", v[j + i * n]);
        }
        printf("\n");
    };
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;        ///< layer function
    printf(" out[%d,%d]", m, n);
    switch(fn) {
    case L_CONV2D: {
        Tensor &f = *in.grad[0];     ///< filter tensor
        Tensor &b = *in.grad[1];     ///< bias tensor
        int ks = f.H();              ///< kerneal size
        printf(" ks=%d f[%d,%d], b[%d,%d]", ks, f.H(), f.W(), b.H(), b.W());
        switch(ks) {
        case 3: {
            dim3 g((n+TILE3-1)/TILE3, (m+TILE3-1)/TILE3);
            k_conv2d<TILE3, 3><<<g, blk>>>(da, f.data, b.data, dc, m, n);
        } break;
        case 5: {
            dim3 g((n+TILE5-1)/TILE5, (m+TILE5-1)/TILE5);
            k_conv2d<TILE5, 5><<<g, blk>>>(da, f.data, b.data, dc, m, n);
        } break;
        default: ERROR("model#conv2d kernel_size=%d not supported\n", ks);
        }
        dump(dc, m, n);
    } break;
    case L_LINEAR:  {                         ///< out = w @ in + b
        Tensor &w = *in.grad[0];  
        Tensor &b = *in.grad[1];
        printf(" w[%d,%d] @ in[%d,%d] + b[%d,%d]", w.H(), w.W(), in.H(), in.W(), b.H(), b.W());
        Tensor::copy(b, out);                 ///< add bias first
        Tensor::gemm(w, in, out, 1.0, 1.0);   ///< out += W * in
        dump(dc, (out.numel+9)/10, 10);                 break;
    } break;
    case L_FLATTEN: Tensor::copy(in, out);              break;
    case L_RELU:    k_relu<<<grd, blk>>>(da, dc, m, n); break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: {
        Tensor &t = *in.grad[0];             ///< tmp tensor
        Tensor::copy(in, t);                 /// * copy content for exp calc
        DU sum = t.map(O_EXP).sum() + DU_EPS;/// * sum all probabilities
        printf(" sum=%.2f ", sum);
        Tensor::mat(O_MUL, t, DU1/sum, out); /// * p / sum(p)
        dump(dc, 1, out.numel);
    } break;
    case L_MAXPOOL:
    case L_AVGPOOL: 
    case L_MINPOOL: {
        switch(p) {                          /// pooling stripe size
        case 2: k_pooling<2><<<grd, blk>>>(da, dc, m, n, fn); break;
        case 3: k_pooling<3><<<grd, blk>>>(da, dc, m, n, fn); break;
        default: ERROR("model#conv2d kernel_size=%d not supported\n", p);
        }
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
Model::_iconv2d(Tensor &in, U16 c, DU bias, U16 *opt) {
    U16 m = opt[0], n = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : int((m-1)/2);      ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation
    U16 h = in.H() - 2 * (p - int(m/2));         ///> output height
    U16 w = in.W() - 2 * (p - int(n/2));         ///> output width
    if (m != n || (m != 3 && m != 5)) {
        ERROR("Model#conv2d f=[%d,%d]? 3x3 and 5x5 supported only.\n", m, n);
        return;
    }
    Tensor *f  = in.grad[0] = &tensor(1, m, n, c).map(O_FILL, DU1); ///> f
    Tensor *df = in.grad[2] = &tensor(1, m, n, c).map(O_FILL, DU0); ///> df
    Tensor *b  = in.grad[1] = &tensor(1, h, w, 1).map(O_FILL, DU0); //bias); ///> b
    Tensor *db = in.grad[3] = &tensor(1, h, w, 1).map(O_FILL, DU0);  ///> db
//    _mmu->random(*f, UNIFORM);                   /// * randomize f
//    Tensor::mat(O_SUB, *f, 0.5, *f);
    
    Tensor &out= tensor(1, h, w, c).map(O_FILL, DU0);  ///> output tensor
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_ilinear(Tensor &in, U16 n, DU bias) {
    U16 m = in.H();
    Tensor *w  = in.grad[0] = &tensor(1, n, m, 1).map(O_FILL, DU1);  ///> w
    Tensor *dw = in.grad[2] = &tensor(1, n, m, 1).map(O_FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &vector(n).map(O_FILL, DU1); //bias);          ///> b
    Tensor *db = in.grad[3] = &vector(n).map(O_FILL, DU0);           ///> db
    Tensor::mat(O_MUL, *w, 0.001, *w);
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
    U16 s[4] = { 1, f, f, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
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
