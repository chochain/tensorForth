/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
__HOST__ const char*
Model::nname(int i) {               ///< network layer name
    static const char *name[] = {   /// double check with t4_layer
    "output ", "conv2d ", "linear ", "flatten", "relu   ",
    "tanh   ", "sigmoid", "softmax", "maxpool", "avgpool",
    "minpool", "dropout"
    };
    return name[i];
}
__GPU__ const char*
Model::d_nname(int i) {
    static const char* name[] = {   /// double check with t4_layer
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
__GPU__ DU
Model::loss(t4_loss op, Tensor &exp) {
    Tensor &out = (Tensor&)_mmu->du2obj(data[numel - 1]);
    if (!out.is_same_shape(exp)) { ERROR("Model::loss dim?\n"); return; }

    Tensor &err = _mmu->copy(out);
    DU     rst  = DU0;
    switch (op) {
    case LOSS_NLL: break;
    case LOSS_MSE: {
        err -= exp;
        Tensor::mat(O_MUL, err, err, err);
        rst = 0.5 * err.sum() / err.numel;
    } break;
    case LOSS_CE:  {
        err.map(O_LOG);
        Tensor::mat(O_MUL, exp, err, err);
        rst = err.sum() / -err.numel;
    } break;
    default: ERROR("Model#loss op=%d not supported\n", op);
    }
    return rst;
}
///
/// Convolution and Linear ops
///
__GPU__ void
Model::_iconv(Tensor &in, U16 C, DU bias, U16 *opt) {
    U16 C1= in.C();
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
    ///
    /// filter: C1 to C fully connected
    /// TODO: filters should have 5th dimension but we steal N for C1 now
    ///
    Tensor *f  = in.grad[0] = &tensor(C1, 1, M, N, C);                  ///> f
    Tensor *df = in.grad[2] = &tensor(C1, 1, M, N, C).map(O_FILL, DU0); ///> df
    Tensor *b  = in.grad[1] = &vector(C).map(O_FILL, bias);             ///> b
    Tensor *db = in.grad[3] = &vector(C).map(O_FILL, DU0);              ///> db
    DU k = DU1 / sqrtf(M * N * C1);              /// * filter default range
    _mmu->random(*f, UNIFORM, -0.5, 2.0 * k);    /// * randomize f [-k ~ k)
    printf("bias=%.2f,  k=%.4f, f.std=%.4f\n", bias, k, f->std());
    for (int i=0; i<M; i++) {
        DU *p = &f->data[i * N];
        for (int j=0; j<N; j++, p++) {
            if (*p < DU0) printf(" -%.3f", -*p);
            else          printf("  %.3f", *p); 
        }
        printf("\n");
    }
    
    Tensor &out= tensor(1, h, w, C);             ///> output tensor
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_ilinear(Tensor &in, U16 C, DU bias) {
    U16 C1 = in.numel;
    Tensor *w  = in.grad[0] = &tensor(1, C, C1, 1);                  ///> w
    Tensor *dw = in.grad[2] = &tensor(1, C, C1, 1).map(O_FILL, DU0); ///> dw
    Tensor *b  = in.grad[1] = &vector(C).map(O_FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &vector(C).map(O_FILL, DU0);           ///> db
    
    DU k = DU1 / sqrtf(C1);                      /// * default weight
    _mmu->random(*w, UNIFORM, -0.5, 2.0 * k);    /// * randomize w
    printf("bias=%.2f,  k=%.4f, w.std=%.4f\n", bias, k, w->std());
    
    Tensor &out = vector(C);                     ///> output tensor sizing
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
    npush(out);                   /// * stage for next stage
}
__GPU__ void
Model::_idropout(Tensor &in, U16 f) {
    Tensor &out = _mmu->copy(in);
    Tensor *msk = in.grad[0] = &_mmu->copy(in);  ///> dropout mask
    
    in.parm = f;                                 /// * keep fraction
    DU p = -0.01 * f;                            ///< dropout fraction
    _mmu->random(*msk, UNIFORM, p);              /// * randomize w, shift p
    printf("dropout=%d\%\n", f);
    
    npush(out);
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
