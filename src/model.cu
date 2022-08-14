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

template<int Z, int K>      /// WARP_SIZE, KERNEL_SIZE
__KERN__ void k_conv2d(
    DU *A, DU *B, DU *C,    ///< input A[MxN], kernel B[KxK], output C[MxN]
    int M, int N
    ) {
    __shared__ DU d[Z * Z];                 ///< shared memory [WARPxWARP]
    
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * blockDim.x;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * blockDim.y;
    const int i  = i0 - int(K / 2);         ///< transfrom to input coordinate
    const int j  = j0 - int(K / 2);
    
    d[tx + ty * Z] =                        /// shared memory with zero padding
        (i >= 0 && i < M && j >= 0 && j < N)
        ? A[j + i * N] : DU0;
    __syncthreads();

    if (i0 < M && j0 < N) {                 /// update C[i][j]
        DU sum = DU0;
        for (int y = 0; y < K; y++) {       /// process one cell
            int d0 = tx + (y + ty) * Z;     ///< offset to smem block
            int b0 = y * K;
            for (int x = 0; x < K; x++) {
                sum += d[x + d0] * B[x + b0];
            }
        }
        C[j0 + i0 * N] = sum;               ///< output matrix
    }
}

__KERN__ void k_pooling(
    DU *A, DU *C,
    int M, int N, int K,
    t4_pool_op op
    ) {
    const int tx = threadIdx.x, j0 = tx + blockIdx.x * blockDim.x;
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * blockDim.y;
    const int j  = j0 * K;
    const int i  = i0 * K;
    
    if (i0 < M && j0 < N) {
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
///
/// Convolution and Linear Layers
///
__GPU__ Model&
Model::step(t4_pool_op op) {
    Tensor &in  = *nten;
    Tensor &out = *(nten+1);
    int    m    = out.H();
    int    n    = out.W();
    dim3   block(WARP_SZ, WARP_SZ);

    auto conv3 = [in, out, m, n, block](DU *kn) {
        dim3 grid((n+CONV_SZ_3-1)/CONV_SZ_3, (m+CONV_SZ_3-1)/CONV_SZ_3);
        k_conv2d<WARP_SZ, 3><<<grid, block>>>(in.data, kn, out.data, m, n);
    };
    auto conv5 = [in, out, m, n, block](DU *kn) {
        dim3 grid((n+CONV_SZ_5-1)/CONV_SZ_5, (m+CONV_SZ_5-1)/CONV_SZ_5);
        k_conv2d<WARP_SZ, 5><<<grid, block>>>(in.data, kn, out.data, m, n);
    };
    auto pool = [in, out, m, n, block, op]() {
        int k = in.parm;
        dim3 grid((n+WARP_SZ-1)/WARP_SZ, (m+WARP_SZ-1)/WARP_SZ);
        k_pooling<<<grid, block>>>(in.data, out.data, m, n, k, op);
    };
    
    Tensor &kn = *in.grad[0];
    switch(in.grad_fn) {
    case DCONV2D: {
        int k = kn.H();
        switch(k) {
        case 3: conv3(kn.data); break;
        case 5: conv5(kn.data); break;
        default: ERROR("model#conv2d kernel size=%d not supported\n", k);
        }
    } break;
    case DLINEAR:  break;
    case DFLATTEN: break;
    case DRELU:    break;
    case DTANH:    break;
    case DSIGMOID: break;
    case DSOFTMAX: break;
    case DMAXPOOL: pool(); break;
    case DAVGPOOL: pool(); break;
    case DMINPOOL: pool(); break;
    case DDROPOUT: break;
    }
    return *this;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
