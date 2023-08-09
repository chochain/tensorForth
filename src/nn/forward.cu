/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network feed forward implementation
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
    __shared__ DU _I[T4_WARP_SQ];                    ///< shared memory [16x16]

    const int tx = threadIdx.x, j0 = tx + blockIdx.x * TS;   ///< output coordinates
    const int ty = threadIdx.y, i0 = ty + blockIdx.y * TS;   /// * i0,j0=0:29
    const int c0 = blockIdx.z,  C0 = gridDim.z;      ///< channel deep
    const int z0 = c0 + (j0 + i0 * W) * C0;          ///< output array index
    const int xy = tx + ty * T4_WARP_SZ;             ///< tile index
    ///
    /// process z0, i.e. [TS, TS, C] cells per kernel call
    ///
    const int i1 = i0 - INT(KS / 2);                 ///< input coordinates
    const int j1 = j0 - INT(KS / 2);                 /// * i1,j1=-1:28

    auto g = cg::this_thread_block();                ///< all threads of block
    for (int c1 = 0; c1 < C1; c1++) {                ///< each input channel
        const int z1 = c1 + (j1 + i1 * W) * C1;      ///< one channel at a time
        _I[xy] =                                     /// * cache input data
            (i1 >= 0 && i1 < H && j1 >= 0 && j1 < W) /// * with zero padding
            ? I[z1] : DU0;                           /// * by channel
        g.sync();                                    /// * smem write barrier
        ///
        /// Y = sum(W * X)
        /// TODO: cache F
        ///
        const int zf = c0 + c1 * KS * KS * C0;       ///< filter index [C1,KS,KS,C0]
        if (tx < TS && ty < TS) {                    /// * each tile
            DU sum = DU0;
            DU *fx = &F[zf], *ix = &_I[xy];          /// * filter[0], tile[tx,ty]
            #pragma unroll
            for (int y = 0; y < KS; y++) {           /// * process one KS * KS cell
                for (int x = 0; x < KS; x++) {
                    sum += (*fx) * ix[x];            /// Y += W * X
                    fx  += C0;                       /// * next filter cell
                }
                ix += T4_WARP_SZ;                    /// next row of tile
            }
            if (i0 < W && j0 < H) {
                if (c1==0) O[z0] = sum + B[c0];      /// * O[ijc] with bias
                else       O[z0] += sum;
            }
        }
        g.sync();
    }
}

__KERN__ void k_linear(
    DU *I, DU *O, DU *W, DU *B,
    int C1, int C0, int HWC1, int HWC0
    ) {
    const int c1 = threadIdx.x + blockIdx.x * blockDim.x;
    const int c0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int n  = blockIdx.z;

    if (c0 < C0 && c1 < C1) {
        /*
        ///* blk.C1=1 ~25% slower but shuffle-sum might get better
        DU *w  = &W[c0 * C1], *x = &I[n * HWC1], acc = B[c0];
        for (int k = 0; k < C1; k++) {
            acc += (*w++) * (*x++);
        }
        O[c0 + n * HWC0] = acc;
        */
        DU *y = &O[c0 + n * HWC0];
        if (c1 == 0) *y = B[c0];                      /// Y = WX + B
        atomicAdd_block(y, W[c1 + c0 * C1] * I[c1 + n * HWC1]);
    }
}

template<int KS>                                      /// kernel size
__KERN__ void k_pool(
    t4_layer op,                                      ///< pooling ops
    DU *I, DU *O,                                     ///< input, output buffers
    int H, int W                                      ///< output HW (C0==C1)
    ) {
    const int HW = H * W;                             ///< output dimension
    const int k0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int j0 = k0 % W;                            ///< output x dim
    const int c  = blockIdx.y, C = gridDim.y;         ///< channel deep
    const int ns = blockIdx.z * HW * C;               ///< batch slice idx
    const int z0 = c + k0 * C + ns;                   ///< output array index
    const int z1 = c + j0 * KS * C + ((k0 - j0) * C + ns) * KS * KS;
    const bool avg = (op != L_MAXPOOL && op != L_MINPOOL);

    if (k0 < HW && c < C) {
        const int RI = (W - 1) * KS * C;              ///< input cell row increment
        DU *ix = &I[z1];
        DU2 v  = avg ? DU0 : *ix;
        #pragma unroll
        for (int y = 0; y < KS; y++) {
            for (int x = 0; x < KS; x++) {
                DU dx = *ix;
                switch (op) {
                case L_USAMPLE:
                case L_AVGPOOL: v += dx;        break;
                case L_MAXPOOL: v = MAX(dx, v); break;
                case L_MINPOOL: v = MIN(dx, v); break;
                }
                ix += C;                             /// * next cell
            }
            ix += RI;                                /// * next row
        }
        O[z0] = avg ? v / (KS * KS) : v;
    }
}

__KERN__ void k_filter(
    DU *I, DU *F, DU *O,                   ///< input, filter, output tensors
    int HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index

    if (i < HW) {
        O[k] = (F[k] > DU0) ? I[k] : DU0;
    }
}

#define SELU_L  1.0507                     /** Selu lambda */
#define SELU_LA 1.7581                     /** Selu alpha  */

__KERN__ void k_activate(
    t4_layer op, DU *I, DU *O,             ///< func, input, output tensors
    DU alpha, int HW                       ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index

    if (i < HW) {
        DU ik = I[k];                                      ///< use register
        switch (op) {
        case L_TANH:
            ik = O[k] = TANH(ik);                       
            I[k] = DU1 - ik*ik;                     break; /// * cache (1 - tanh^2)
        case L_SIGMOID:
            ik = O[k] = SIGMOID(ik);
            I[k] = ik * (DU1 - ik);                 break; /// * cache sig*(1 - sig)
        case L_SELU:    O[k] = ik > DU0                    /// * cache selu in I[k]
            ? (I[k] = SELU_L, ik)
            : (I[k] = SELU_LA * EXP(ik)) - SELU_LA; break;
        case L_LEAKYRL: O[k] = ik > DU0
            ? (I[k] = DU1, ik)
            : (I[k] = alpha) * ik;                  break;
        case L_ELU:     O[k] = ik > DU0
            ? (I[k] = DU1, ik)
            : (I[k] = alpha * EXP(ik)) - alpha;     break;
        }
    }
}

__KERN__ void k_batchnorm(
    DU *I, DU *O,  DU *X,                  ///< input, filter, output tensors
    DU *avg, DU *ivar,                     ///< mean, gamma/(stdvar - e)
    DU *gamma, DU *beta,
    int HW                                 ///< H0=H1, W0==W1 (C0==C1)
    ) {
    const int i  = threadIdx.x + blockIdx.x * blockDim.x;  ///< element index
    const int c  = blockIdx.y, C = gridDim.y;              ///< channel deep
    const int ns = blockIdx.z * HW * C;                    ///< batch slice index
    const int k  = c + i * C + ns;                         ///< output tensor index

    if (i < HW) {
        O[k] = (X[k] = (I[k] - avg[c]) * ivar[c]) * gamma[c] + beta[c];
    }
}
//
//< Neaural network forward propagation
// * input can be a Tensor or a Dataset
//
__GPU__ Model&
Model::forward(Tensor &input) {
    Tensor &n1 = (*this)[1];  ///< reference model input layer

    if (input.numel != n1.numel) {
        ERROR("Model::forward dataset wrong shape[%d,%d,%d,%d] != model input[[%d,%d,%d,%d]\n",
            input.N(), input.H(), input.W(), input.C(),
            n1.N(), n1.H(), n1.W(), n1.C());
        return *this;
    }
    n1 = input;               /// * copy dataset batch into the first layer
    ///
    /// cascade execution layer by layer forward
    /// TODO: model execution becomes a superscalar pipeline
    ///
    auto trace = [](DU t, int i, Tensor &in, Tensor &out) {
        printf("\n%6.2f:%2d> %s Σ/n=%6.2f [%d,%d,%d,%d]\tp=%-2d => out[%d,%d,%d,%d]",
            t, i, d_nname(in.grad_fn), in.sum() / in.N() / in.C(),
            in.N(), in.H(), in.W(), in.C(), in.parm,
            out.N(), out.H(), out.W(), out.C());
    };
    int tlvl = _mmu->trace();
    TRACE1("\nModel::forward starts");
    DU t0 = _mmu->ms(), t1 = t0, tt;             ///< performance measurement
    for (U16 i = 1; i < numel - 1; i++) {
        Tensor &in = (*this)[i], &out = (*this)[i + 1];
        if (tlvl) {
            trace((tt=_mmu->ms()) - t1, i, in, out);
            t1 = tt;
        }
        _fstep(in, out);
        if (tlvl) debug(out);
    }
    ///
    /// collect onehot vector and hit count
    ///
    if (input.is_dataset()) {
        if (_hot) _mmu->free(*_hot);
        _hot = &onehot((Dataset&)input);         /// * create/cache onehot vector
        _hit = hit(true);                        /// * recalc/cache hit count
    }
    TRACE1("\nModel::forward %5.2f ms\n", _mmu->ms() - t0);

    return *this;
}
/// ========================================================================
/// private methods
///
__GPU__ void
Model::_fstep(Tensor &in, Tensor &out) {
    ///
    /// layer function dispatcher
    ///
    t4_layer fn = in.grad_fn;                       ///< layer function
    switch(fn) {
    case L_CONV:    _fconv(in, out);         break; ///< convolution
    case L_LINEAR:  _flinear(in, out);       break; ///< out = W @ in + B
    case L_FLATTEN: out = in;                break; ///< straight copy
    case L_RELU:    _ffilter(in, in, out);   break; ///< filter in < 0
    case L_TANH:
    case L_SIGMOID:
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:     _factivate(in, out, fn); break;
    case L_SOFTMAX: _fsoftmax(in, out);      break; /// * feed to CrossEtropy
    case L_LOGSMAX: _flogsoftmax(in, out);   break; /// * feed to NLL
    case L_AVGPOOL:
    case L_MAXPOOL:
    case L_MINPOOL: _fpool(in, out, fn);     break;
    case L_DROPOUT: {                               ///< dropout mask
        DU     pct  = 0.001 * in.parm;              ///< percentage dropout
        Tensor &msk = *in.grad[0];                  ///< dropout mask
        _mmu->random(msk, UNIFORM, -pct);           /// * randomize w, shift pct
        _ffilter(in, msk, out);
    } break;
    case L_USAMPLE: _fupsample(in, out, fn); break;
    case L_BATCHNM: _fbatchnorm(in, out);    break;
    default: ERROR("Model::forward layer=%d not supported\n", fn);
    }
}

#define TILE1    (T4_WARP_SZ)              /** 16, 1x1 conv */
#define TILE3    (T4_WARP_SZ - 3 + 1)      /** 14, 3x3 conv */
#define TILE5    (T4_WARP_SZ - 5 + 1)      /** 12, 5x5 conv */

__GPU__ int
Model::_fconv(Tensor &in, Tensor &out) {
    Tensor &tf = *in.grad[0];                             ///< filter tensor
    Tensor &tb = *in.grad[1];                             ///< bias tensor

    TRACE1(" f[%d,%d,%d,%d], b[%d]", tf.N(), tf.H(), tf.W(), tf.C(), tb.numel);

    const int N = out.N(), H = out.H(), W = out.W();      ///< outpt dimensions
    const int C0 = out.C(), C1 = in.C();                  ///< output, input channel deep

    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);                  ///< default blocks
    dim3 g1((W + TILE1 - 1) / TILE1, (H + TILE1 - 1) / TILE1, C0);
    dim3 g3((W + TILE3 - 1) / TILE3, (H + TILE3 - 1) / TILE3, C0);
    dim3 g5((W + TILE5 - 1) / TILE5, (H + TILE5 - 1) / TILE5, C0);

    for (int n = 0; n < N; n++) {
        DU *d1 = in.slice(n), *d0 = out.slice(n);
        DU *f  = tf.data, *b = tb.data;
        int ks = tf.H();
        switch(ks) {                       /// * TODO: handles rectangular filters
        case 1: k_conv2d<TILE1,1><<<g1,blk>>>(d1, f, b, d0, H, W, C1); break;
        case 3: k_conv2d<TILE3,3><<<g3,blk>>>(d1, f, b, d0, H, W, C1); break;
        case 5: k_conv2d<TILE5,5><<<g5,blk>>>(d1, f, b, d0, H, W, C1); break;
        default:
            ERROR("model_fwd#conv kernel_size=%d not supported\n", ks);
            return -1;
        }
        GPU_SYNC();
    }
    return 0;
}

__GPU__ int
Model::_flinear(Tensor &in, Tensor &out) {
    auto qa_calc = [&in, &out](DU *w, DU *b, int N, int C1, int C0) {
        for (int n = 0; n < N; n++) {                 /// * walk through batch
            DU *x = in.slice(n), *y = out.slice(n);
            for (int c0 = 0; c0 < C0; c0++) {
                y[c0] = b[c0];                        /// init with bias
                for (int c1 = 0; c1 < C1; c1++) {     /// dot product
                    y[c0] += w[c1 + c0 * C1] * x[c1]; /// Y = W @ X + B
                }
            }
        }
    };
    Tensor &tw = *in.grad[0];                         ///< weight tensor
    Tensor &tb = *in.grad[1];                         ///< bias tensor

    const int N  = out.N();                           ///< batch size (N1 == N0)
    const int C0 = tw.H(), C1 = tw.W();               ///< dense layer dims

    TRACE1(" = w[%d,%d] @ in[%d,%d,%d,%d] + b[%d]",
        C0, C1, in.N(), in.H(), in.W(), in.C(), tb.numel);

    if (tw.numel > T4_WARP_SQ) {                      /// * threadhold control
        dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);          ///< default blocks
        dim3 grd(NGRID(C1, C0, N, blk));              ///< default grids

        k_linear<<<grd,blk>>>(
            in.data, out.data, tw.data, tb.data, C1, C0, in.HWC(), out.HWC());
        GPU_SYNC();                                   /// * this makes it slow
    }
    else qa_calc(tw.data, tb.data, N, C1, C0);        /// * serial code (for validation)
    // _dump(out.data, out.H(), out.W(), out.C());
    return 0;
}

__GPU__ int
Model::_ffilter(Tensor &in, Tensor &msk, Tensor &out) {
    const int W = out.W(), HW = out.H() * W;

    dim3 blk(T4_WARP_SQ, 1, 1);                        ///< default blocks
    dim3 grd((HW + blk.x - 1)/blk.x, out.C(), out.N());

    k_filter<<<grd, blk>>>(in.data, msk.data, out.data, HW);
    GPU_SYNC();

    return 0;
}

__GPU__ int
Model::_factivate(Tensor &in, Tensor &out, t4_layer fn) {
    const int W = out.W(), HW = out.H() * W;

    dim3 blk(T4_WARP_SQ, 1, 1);                        ///< default blocks
    dim3 grd((HW + blk.x - 1)/blk.x, out.C(), out.N());

    DU alpha = 0.001 * in.parm;

    k_activate<<<grd, blk>>>(fn, in.data, out.data, alpha, HW);
    GPU_SYNC();

    return 0;
}

__GPU__ int
Model::_fpool(Tensor &in, Tensor &out, t4_layer fn) {
    const int W  = out.W(), H = out.H();                ///< output dimensions
    const int ks = in.parm;                             ///< kernel size

    dim3 blk(T4_WARP_SQ, 1, 1);                         ///< default blocks
    dim3 grd((H * W + blk.x - 1) / blk.x, out.C(), out.N());

    switch(ks) {                                        /// pooling kernel size
    case 2: k_pool<2><<<grd,blk>>>(fn, in.data, out.data, H, W); break;
    case 3: k_pool<3><<<grd,blk>>>(fn, in.data, out.data, H, W); break;
    default:
        ERROR("model#pooling kernel_size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();

    return 0;
}

__GPU__ int
Model::_fsoftmax(Tensor &in, Tensor &out) {
    out = in;                                   /// copy content for exe calc
    out.map(O_EXP);                             /// *
    Tensor &t = _t4(1, in.H(), in.W(), in.C()); ///< create temp tensor for calc
    DU     *d = t.data;                         ///< cached tensor data
    for (int n = 0; n < in.N(); n++) {          ///< loop thru mini-batch
        t.data = out.slice(n);                  /// * point to output data slice
        DU sum = t.sum();                       ///< sum(exp(xi))
        t.map(O_MUL, DU1 / (sum + DU_EPS));     /// * softmax = exp(xi)/sum(exp(xi))
    }
    t.data = d;                                 /// * restore tensor data
    _mmu->free(t);                              /// * release memory
    return 0;
}

__GPU__ int
Model::_flogsoftmax(Tensor &in, Tensor &out) {  /// * TODO: DCP
    out = in;                                   /// * copy in data to out
    out.map(O_EXP);
    Tensor &t = _t4(1, in.H(), in.W(), in.C()); ///< create tmp tensor
    DU     *d = t.data;                         ///< cache tensor data
    for (int n = 0; n < in.N(); n++) {          /// * loop throught mini-batch
        t.data = out.slice(n);
        DU sum    = t.sum();
        DU logsum = LOG(sum > DU0 ? sum : DU_EPS);
        t -= logsum;                            ///< xi - log(sum(exp(xi)))
    }
    t.data = d;                                 /// * restore tensor data pointer
    _mmu->free(t);                              /// * release memory
    return 0;
}
///
///> upsampling =~ reverse pooling (calls backprop k_dpool)
///
template<int KS>                                        /// forward declare (in backprop.cu)
__KERN__ void k_dpool(t4_layer op, DU *I, DU *O, int H, int W);
__GPU__ int
Model::_fupsample(Tensor &in, Tensor &out, t4_layer fn) {
    const int W  = in.W(), H = in.H();                  ///< input dimensions (reversed pool)
    const int me = (in.parm >> 8);                      ///< upsample method, TODO
    const int ks = in.parm & 0xff;                      ///< upsampling size

    dim3 blk(T4_WARP_SQ, 1, 1);
    dim3 grd((H * W + blk.x - 1) / blk.x, in.C(), in.N());

    switch(ks) {
    case 2: k_dpool<2><<<grd,blk>>>(fn, out.data, in.data, H, W); break;
    case 3: k_dpool<3><<<grd,blk>>>(fn, out.data, in.data, H, W); break;
    default:
        ERROR("model#upsample size=%d not supported\n", ks);
        return -1;
    }
    GPU_SYNC();

    //_dump(in.data,  in.H(), in.W(), in.C());
    //_dump(out.data, out.H(), out.W(), out.C());
    return 0;
}

///
///> batch norm
///  Note: borrow k_sum, k_var from ~/mmu/tensor.cu
///
extern __KERN__ void k_sum(DU *I, DU *sum, int HW);
extern __KERN__ void k_var(DU *I, DU *avg, DU *var, int HW);
__GPU__ int
Model::_fbatchnorm(Tensor &in, Tensor &out) {
    const int W = out.W(), HW = out.H() * W;
    const int C = out.C(), NHW = HW * out.N();         ///< C0==C1

    dim3 blk(T4_WARP_SQ, 1, 1);                        ///< default blocks
    dim3 grd((HW + blk.x - 1)/blk.x, C, out.N());

    DU *w   = &in.grad[0]->data[0];                    ///< weight/gamma
    DU *b   = &in.grad[0]->data[C];                    ///< bias/beta
    DU *avg = &in.grad[1]->data[0];                    ///< mean
    DU *var = &in.grad[1]->data[C];                    ///< 1.0/(var+e)^0.5
    DU *xht = in.grad[3]->data;                        ///< x_hat

    for (int c=0; c < C; c++) avg[c] = var[c] = DU0;   /// * zero
    k_sum<<<grd, blk>>>(in.data, avg, HW);             /// * capture sum
    GPU_SYNC();

    for (int c=0; c < C; c++) avg[c] /= NHW;           /// * calc mean per channel
    k_var<<<grd, blk>>>(in.data, avg, var, HW);        /// * capture variance
    GPU_SYNC();

    const DU m = 0.001 * in.parm;                      ///< ETA momentum, TODO:
    for (int c=0; c < C; c++) {
        var[c] = 1.0 / SQRT(var[c] / NHW + DU_EPS);    ///< calc population stdvar
    }

    k_batchnorm<<<grd, blk>>>(                         /// * O = x_hat*gamma + beta
        in.data, out.data, xht, avg, var, w, b, HW
    );
    GPU_SYNC();

    return 0;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
