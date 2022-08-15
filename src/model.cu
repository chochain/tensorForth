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
                sum += d[x + d0] * F[x + b0] + B[x + b0]; 
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
/// Convolution and Linear ops
///
__GPU__ Model&
Model::iconv2d(DU bias, U16 c, U16 *opt) {
    if (NO_INIT) return *this;
    Tensor &in = *nten; in.grad_fn = L_CONV2D;   ///> derivative function

    U16 m = opt[0], n = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : int((m-1)/2);      ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation

    Tensor *f  = in.grad[0] = &tensor(1, m, n, c);                   ///> f
    Tensor *b  = in.grad[1] = &tensor(1, m, n, 1).map(O_FILL, bias); ///> b
    Tensor *df = in.grad[2] = &tensor(1, m, n, c).map(O_FILL, DU0);  ///> df
    Tensor *db = in.grad[3] = &tensor(1, m, n, 1).map(O_FILL, DU0);  ///> db
    _mmu->random(*f, NORMAL);                    /// * randomize f
    
    Tensor &out = tensor(                        ///> output tensor sizing
        1,
        in.H() + 2 * (p - int(m/2)),
        in.W() + 2 * (p - int(n/2)),
        c).map(O_FILL, DU0);
    npush(out);                                  /// * stage for next stage
    return *this;
}
__GPU__ Model&
Model::ilinear(DU bias, U16 n) {
    if (NO_INIT) return *this;
    Tensor &in = *nten; in.grad_fn = L_LINEAR;   ///> derivative function

    U16 m = in.H();
    Tensor *w  = in.grad[0] = &tensor(1, n, m, 1);                   ///> w
    Tensor *dw = in.grad[2] = &tensor(1, n, m, 1).map(O_FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &vector(n).map(O_FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &vector(n).map(O_FILL, DU0);           ///> db
    _mmu->random(*w, NORMAL);                    /// * randomize w
    
    Tensor &out = vector(n);                     ///> output tensor sizing
    npush(out);                                  /// * stage for next stage
    return *this;
}
__GPU__ Model&
Model::iflatten() {
    if (NO_INIT) return *this;
    Tensor &in  = *nten;
    Tensor &out = vector(in.numel);
    in.grad_fn  = L_FLATTEN;
    in.parm     = in.numel;
    npush(out);
    return *this;
}
///
/// Activation ops
///
__GPU__ Model&
Model::irelu() {
    if (NO_INIT) return *this;
    Tensor &in  = *nten;
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    in.grad_fn  = L_RELU;
    npush(out);                   /// * stage for next stage
    return *this;
}
__GPU__ Model&
Model::itanh() {
    return *this;
}
__GPU__ Model&
Model::isigmoid() {
    return *this;
}
__GPU__ Model&
Model::isoftmax() {
    if (NO_INIT) return;
    Tensor &in  = *nten;
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    in.grad_fn  = L_SOFTMAX;
    npush(out);                   /// * stage for next stage
    return *this;
}
///
/// Pooling and Dropout ops
///
__GPU__ Model&
Model::imaxpool(U16 f) {
    if (NO_INIT) return *this;
    Tensor &in = *nten; in.grad_fn = L_MAXPOOL;
    in.parm    = f;              /// * keep pooling width
    
    U16 m = int((in.H() - f) / f) + 1;
    U16 n = int((in.W() - f) / f) + 1;
    U16 s[4] = { 1, f, f, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
    Tensor &out = tensor(1, m, n, in.C());
    npush(out);                 /// * stage for next stage
    return *this;
}
__GPU__ Model&
Model::iavgpool(U16 n) {
    return *this;
}
__GPU__ Model&
Model::iminpool(U16 n) {
    return *this;
}
__GPU__ Model&
Model::idropout(U16 f) {
    if (NO_INIT) return *this;
    Tensor &in  = *nten;
    Tensor &out = _mmu->copy(in);
    in.grad_fn  = L_DROPOUT;
    in.parm     = f;
    npush(out);
    return *this;
}
///
/// private methods
///
__GPU__ void
Model::step(Tensor &in, Tensor &out) {
    DU     *da  = in.data;
    DU     *dc  = out.data;
    int    m    = out.H();
    int    n    = out.W();
    int    k    = in.parm;
    dim3   blk(WARP_SZ, WARP_SZ);
    dim3   grd((n + WARP_SZ - 1)/WARP_SZ, (m + WARP_SZ - 1)/WARP_SZ);

    auto conv3 = [da, dc, m, n, blk](DU *f, DU *b) {
        dim3 g((n+CONV_SZ_3-1)/CONV_SZ_3, (m+CONV_SZ_3-1)/CONV_SZ_3);
        k_conv2d<WARP_SZ, CONV_SZ_3, 3><<<g, blk>>>(da, f, b, dc, m, n);
    };
    auto conv5 = [da, dc, m, n, blk](DU *f, DU *b) {
        dim3 g((n+CONV_SZ_5-1)/CONV_SZ_5, (m+CONV_SZ_5-1)/CONV_SZ_5);
        k_conv2d<WARP_SZ, CONV_SZ_5, 5><<<g, blk>>>(da, f, b, dc, m, n);
    };
    
    switch(in.grad_fn) {
    case L_CONV2D: {
        Tensor &f = *in.grad[0];     ///< filter tensor
        Tensor &b = *in.grad[1];     ///< bias tensor
        int k = f.H();
        switch(k) {
        case 3: conv3(f.data, b.data); break;
        case 5: conv5(f.data, b.data); break;
        default: ERROR("model#conv2d kernel size=%d not supported\n", k);
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
#endif  // T4_ENABLE_OBJ
//==========================================================================
