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
template<int Th, int Tw>
__KERN__ void k_conv_row(
    DU *A, DU *B, DU *C,    /// input A[MxN], kernel B[KxK], output C[MxN]
    int M, int N, int K
    ) {
    __shared__ DU data[Th * (Tw + K * 2) ];
    
    const int x   = threadIdx.x, y = threadIdx.y;
    const int j   = x + blockIdx.x * blockDim.x;
    const int i   = y + blockIdx.y * blockDim.y;
    const int off = y * (Tw + K * 2);
    const int idx = j + i * N;
    /// padding
    data[x + off]              = ((j - K) < 0) ? DU0 : A[idx - K];
    data[x + blockDim.x + off] = ((j + K) < N) ? A[idx + K] : DU0;
    __syncthreads();

    DU sum = DU0;
    for (int k = -K, j1 = x + K; k <= K; k++) {
        sum += data[j1 + k + off] * B[k + K];
    }
    C[idx] = sum;
}

template<int Th, int Tw>
__KERN__ void k_conv_col(
    DU *A, DU *B, DU *C,    /// input A[MxN], kernel B[KxK], output C[MxN]
    int M, int N, int K
    ) {
    __shared__ DU data[Tw * (Th + K * 2)];
    const int x    = threadIdx.x, y = threadIdx.y;
    const int j    = x + blockIdx.x * blockDim.x;
    const int i    = y + blockIdx.y * blockDim.y;
    const int off  = y * Tw;
    const int idx  = j + i * N;

    /// padding
    data[x + off]                   = ((i - K) < 0) ? DU0 : A[idx - K * N];
    data[x + blockDim.y * Tw + off] = ((i + K) < M) ? A[idx + K * N] : DU0;
    __syncthreads();

    DU sum = DU0;
    for (int k = 0, x1 = x + y * Tw; k <= K*2; k++) {
        sum += data[x1 + k * Tw] * B[k];
    }
    C[idx] = sum;
}

typedef enum {
    POOL_MAX = 0,
    POOL_MIN,
    POOL_AVG
} t4_pool_op;

__KERN__ void k_pooling(
    DU *A, DU *B, DU *C,
    int M, int N, int K,
    DU alpha, DU beta,
    t4_pool_op
    ) {
    int x = threadIdx.x, y = threadIdx.y;
    int i = (y + blockIdx.y * blockDim.y) * K;
    int j = (x + blockIdx.x * blockDim.x) * K;

    if (i < M && j < N) {
        DU2 acc = 0;
        for (int k = 0; k < K; ++k) {
            acc += A[k + i * K] * B[j + k * N];
        }
        C[j + i * N] = alpha * acc + beta * C[j + i * N];
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
Model::iconv2d(DU bias, IU c, U16 *opt) {
    if (NO_INIT) return *this;
    Tensor &in = *nten; in.grad_fn = DCONV2D;    ///> derivative function

    U16 m = opt[0], n = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : int((m-1)/2);      ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation

    Tensor *w  = in.grad[0] = &tensor(1, m, n, c);                 ///> w
    Tensor *b  = in.grad[1] = &tensor(1, m, n, 1).map(FILL, bias); ///> b
    Tensor *dw = in.grad[2] = &tensor(1, m, n, c).map(FILL, DU0);  ///> dw
    Tensor *db = in.grad[3] = &tensor(1, m, n, 1).map(FILL, DU0);  ///> db
    _mmu->random(*w, NORMAL);                    /// * randomize w
    
    Tensor &out = tensor(                        ///> output tensor sizing
        1,
        in.H() + 2 * (p - int(m/2)),
        in.W() + 2 * (p - int(n/2)),
        c).map(FILL, DU0);
    npush(out);                                  /// * stage for next stage
    return *this;
}
__GPU__ Model&
Model::ilinear(DU bias, U16 n) {
    if (NO_INIT) return *this;
    Tensor &in = *nten; in.grad_fn = DLINEAR;    ///> derivative function

    U16 m = in.H();
    Tensor *w  = in.grad[0] = &tensor(1, n, m, 1);                 ///> w
    Tensor *dw = in.grad[2] = &tensor(1, n, m, 1).map(FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &vector(n).map(FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &vector(n).map(FILL, DU0);           ///> db
    _mmu->random(*w, NORMAL);                    /// * randomize w
    
    Tensor &out = vector(n);                     ///> output tensor sizing
    npush(out);                                  /// * stage for next stage
    return *this;
}
__GPU__ Model&
Model::iflatten() {
    if (NO_INIT) return *this;
    Tensor &in  = *nten;
    Tensor &out = vector(in.size);
    in.grad_fn  = DFLATTEN;
    in.parm     = in.size;
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
    in.grad_fn  = DRELU;
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
    in.grad_fn  = DSOFTMAX;
    npush(out);                   /// * stage for next stage
    return *this;
}
///
/// Pooling and Dropout ops
///
__GPU__ Model&
Model::imaxpool(U16 f) {
    if (NO_INIT) return *this;
    Tensor &in  = *nten; in.grad_fn = DMAXPOOL;
    in.parm     = f;             /// * keep pooling width
    
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
    in.grad_fn  = DDROPOUT;
    in.parm     = f;
    npush(out);
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
