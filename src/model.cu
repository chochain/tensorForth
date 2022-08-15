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
#define CONV_SZ_3    (WARP_SZ - 3 + 1)
#define CONV_SZ_5    (WARP_SZ - 5 + 1)

template<int Z, int T, int K>    ///> WARP_SIZE, TILE_SIZE, KERNEL_SIZE
__KERN__ void k_conv2d(          ///< TODO: C
    DU *A, DU *F, DU *B, DU *C,  ///> input A[MxN], F[KxK] kernel B[KxK] bias, output C[MxN]
    int M, int N
    ) {
    __shared__ DU d[Z * Z];                 ///< shared memory [WARPxWARP]
    
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * T;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * T;
    const int i  = i0 - int(K / 2);         ///< transfrom to input coordinate
    const int j  = j0 - int(K / 2);
    
    d[tx + ty * Z] =                        ///< shared memory with zero padding
        (i >= 0 && i < M && j >= 0 && j < N)
        ? A[j + i * N] : DU0;
    __syncthreads();

    if (tx < T && ty < T) {                 /// * within tile
        DU sum = DU0;
        for (int y = 0; y < K; y++) {       /// * process one cell
            int d0 = tx + (y + ty) * Z;     ///< offset to smem block
            int b0 = y * K;
            for (int x = 0; x < K; x++) {
                sum += F[x + b0] * d[x + d0] + B[x]; 
            }
        }
        if (i0 < M && j0 < N) {             /// * update C[i][j]
            C[j0 + i0 * N] = sum;           /// * output matrix
        }
    }
}

__KERN__ void k_linear(                     ///< TODO: C
    DU *W, DU *A, DU *B, DU *C,
    int M, int N
    ) {
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < M) {
        DU2 acc = DU0;
        for (int j = 0; j < N; j++) {       ///< TODO: CDP
            acc += W[j + i * N] * A[j];
        }
        C[i] = acc + B[i];
    }
}

__KERN__ void k_pooling(                    ///< TODO: C
    DU *A, DU *C,
    int M, int N, int K,
    t4_pool_op op
    ) {
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * blockDim.x;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * blockDim.y;
    const int j  = j0 * K;
    const int i  = i0 * K;
    
    if (i0 < M && j0 < N) {                ///< TODO: CDP
        DU2 v = (op==POOL_AVG) ? DU0 : A[j + i * N];
        for (int y = 0; y < K; y++) {
            DU *d = &A[j + (y + i) * N];
            for (int x = 0; x < K; x++, d++) {
                switch (op) {
                case POOL_MAX: if (*d > v) v = *d; break;
                case POOL_AVG: v += *d;            break;
                case POOL_MIN: if (*d < v) v = *d; break;
                }
            }
        }
        C[j0 + i0 * N] = (op==POOL_AVG) ? v/K/K : v;
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
    Tensor &in = _mmu->du2ten(data[numel - 1]);
    if (!autograd || in.grad_fn != L_NONE) return *this;
    
    switch(fn) {
    case L_CONV2D:  _iconv2d(in, n, bias, opt); break;
    case L_LINEAR:  _ilinear(in, n, bias);      break;
    case L_FLATTEN: _iflatten(in);              break;
    case L_RELU:    _irelu(in);                 break;
    case L_TANH:    _itanh(in);                 break;
    case L_SIGMOID: _isigmoid(in);              break;
    case L_SOFTMAX: _isoftmax(in);              break;
    case L_MAXPOOL: _imaxpool(in, n);           break;
    case L_AVGPOOL: _iavgpool(in, n);           break;
    case L_MINPOOL: _iminpool(in, n);           break;
    case L_DROPOUT: _idropout(in, n);           break;
    }
    in.grad_fn = fn;
    return *this;
}

__GPU__ Model&
Model::forward(Tensor &input) {
    Tensor &in = (*this)[1];
    if (!in.is_same_shape(input)) {
        ERROR("model#forward dim?\n");
        return *this;
    }
    Tensor::copy(input, in);
    /*
    for (int i = 2; i < (model.numel - 1); i++) {
        Tensor &out = model[i];
        model.forward(in, out);
        in = out;
    }
    */
    return *this;
}
__GPU__ Model&
Model::backprop(Tensor &output) {
    return *this;
}
/// ========================================================================
/// private methods 
///
__GPU__ void
Model::_step(Tensor &in, Tensor &out) {
    DU   *da = in.data;
    DU   *dc = out.data;
    int  m   = out.H();
    int  n   = out.W();
    int  k   = in.parm;
    dim3 blk(WARP_SZ, WARP_SZ);
    dim3 grd((n + WARP_SZ - 1)/WARP_SZ, (m + WARP_SZ - 1)/WARP_SZ);

    auto conv3x3 = [da, dc, m, n, blk](DU *f, DU *b) {
        dim3 g((n+CONV_SZ_3-1)/CONV_SZ_3, (m+CONV_SZ_3-1)/CONV_SZ_3);
        k_conv2d<WARP_SZ, CONV_SZ_3, 3><<<g, blk>>>(da, f, b, dc, m, n);
    };
    auto conv5x5 = [da, dc, m, n, blk](DU *f, DU *b) {
        dim3 g((n+CONV_SZ_5-1)/CONV_SZ_5, (m+CONV_SZ_5-1)/CONV_SZ_5);
        k_conv2d<WARP_SZ, CONV_SZ_5, 5><<<g, blk>>>(da, f, b, dc, m, n);
    };
    
    switch(in.grad_fn) {
    case L_CONV2D: {
        Tensor &f = *in.grad[0];     ///< filter tensor
        Tensor &b = *in.grad[1];     ///< bias tensor
        int ks = f.H();
        switch(ks) {
        case 3: conv3x3(f.data, b.data); break;
        case 5: conv5x5(f.data, b.data); break;
        default: ERROR("model#conv2d kernel_size=%d not supported\n", ks);
        }
    } break;
    case L_LINEAR:  {                ///< dc = W * da + B
        Tensor &w = *in.grad[0];  
        Tensor &b = *in.grad[1];
        int    W2   = WARP_SZ * WARP_SZ;
        dim3   blk1(1, W2), grd1(1, (w.H() + W2 - 1) / W2);
        k_linear<<<grd1, blk1>>>(w.data, da, b.data, dc, w.H(), w.W());
    } break;
    case L_FLATTEN: out.reshape(out.numel); break;
    case L_RELU:    k_relu<<<grd, blk>>>(da, dc, m, n); break;
    case L_TANH:    break;
    case L_SIGMOID: break;
    case L_SOFTMAX: {
        Tensor &tmp = _mmu->copy(in);
        DU sum = tmp.map(O_EXP).sum();        /// * sum all probabilities
        Tensor::mat(O_DIV, tmp, sum, out);    /// * p / sum(p)
    } break;
    case L_MAXPOOL: k_pooling<<<grd, blk>>>(da, dc, m, n, k, POOL_MAX); break;
    case L_AVGPOOL: k_pooling<<<grd, blk>>>(da, dc, m, n, k, POOL_AVG); break;
    case L_MINPOOL: k_pooling<<<grd, blk>>>(da, dc, m, n, k, POOL_MIN); break;
    case L_DROPOUT: break;
    }
}
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

    Tensor *f  = in.grad[0] = &tensor(1, m, n, c);                   ///> f
    Tensor *df = in.grad[2] = &tensor(1, m, n, c).map(O_FILL, DU0);  ///> df
    Tensor *b  = in.grad[1] = &tensor(1, h, w, 1).map(O_FILL, bias); ///> b
    Tensor *db = in.grad[3] = &tensor(1, h, w, 1).map(O_FILL, DU0);  ///> db
    _mmu->random(*f, NORMAL);                    /// * randomize f
    
    Tensor &out= tensor(1, h, w, c).map(O_FILL, DU0);  ///> output tensor
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_ilinear(Tensor &in, U16 n, DU bias) {
    U16 m = in.H();
    Tensor *w  = in.grad[0] = &tensor(1, n, m, 1);                   ///> w
    Tensor *dw = in.grad[2] = &tensor(1, n, m, 1).map(O_FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &vector(n).map(O_FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &vector(n).map(O_FILL, DU0);           ///> db
    _mmu->random(*w, NORMAL);                    /// * randomize w
    
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
Model::_irelu(Tensor &in) {
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    npush(out);                   /// * stage for next stage
}
__GPU__ void
Model::_itanh(Tensor &in) {}

__GPU__ void
Model::_isigmoid(Tensor &in) {}

__GPU__ void
Model::_isoftmax(Tensor &in) {
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    npush(out);                   /// * stage for next stage
}
///
/// Pooling and Dropout ops
///
__GPU__ void
Model::_imaxpool(Tensor &in, U16 f) {
    in.parm = f;                  /// * keep pooling width
    U16 m = int((in.H() - f) / f) + 1;
    U16 n = int((in.W() - f) / f) + 1;
    U16 s[4] = { 1, f, f, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
    Tensor &out = tensor(1, m, n, in.C());
    npush(out);                 /// * stage for next stage
}
__GPU__ void
Model::_iavgpool(Tensor &in, U16 n) {}

__GPU__ void
Model::_iminpool(Tensor &in, U16 n) {}

__GPU__ void
Model::_idropout(Tensor &in, U16 f) {
    Tensor &out = _mmu->copy(in);
    in.parm = f;
    npush(out);
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
