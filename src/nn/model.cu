/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network model constructor implementation
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
    case L_MINPOOL: _ipool(in, n);              break;
    case L_DROPOUT: _idropout(in, n);           break;
    }
    in.grad_fn = fn;
    return *this;
}
///
/// Convolution and Linear ops
///
__GPU__ void
Model::_iconv(Tensor &in, U16 C0, DU bias, U16 *opt) {
    U16 C1 = in.C();
    U16 M  = opt[0], N = opt[1];                  ///> filter sizing
    U16 p  = opt[2] ? opt[2] : int((M-1)/2);      ///> padding
    U16 s  = opt[3], d = opt[4];                  ///> stride, dilation
    U16 H0 = (in.H() - M + p*2) / s + 1;          ///> output height
    U16 W0 = (in.W() - N + p*2) / s + 1;          ///> output width
    if (M != N || (M != 3 && M != 5)) {
        ERROR("Model#conv2d f=[%d,%d]? 3x3 and 5x5 supported only.\n", M, N);
        return;
    }
    in.stride[0] = in.stride[1] = s;
    ///
    /// filter: C1 to C fully connected
    /// TODO: filters should have 5th dimension but we steal N for C1 now
    ///
    Tensor *f  = in.grad[0] = &tensor(C1, 1, M, N, C0);                  ///> f
    Tensor *df = in.grad[2] = &tensor(C1, 1, M, N, C0).map(O_FILL, DU0); ///> df
    Tensor *b  = in.grad[1] = &vector(C0).map(O_FILL, bias);             ///> b
    Tensor *db = in.grad[3] = &vector(C0).map(O_FILL, DU0);              ///> db

    DU k = DU1 / SQRT(M * N * C1);               /// * filter default range
    _mmu->random(*f, UNIFORM, -0.5, 2.0 * k);    /// * randomize f [-k ~ k)
    /*
    printf("bias=%4.2f,  k=%6.4f, f.std=%6.4f\n", bias, k, f->std());
    for (int i=0; i<f->numel; i++) {
        DU dx = f->data[i];
        printf("%6.3f", dx);
    }
    */
    Tensor &out= tensor(1, H0, W0, C0);          ///> output tensor
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_ilinear(Tensor &in, U16 C0, DU bias) {
    U16 C1 = in.numel;
    Tensor *w  = in.grad[0] = &tensor(1, C0, C1, 1);                  ///> w
    Tensor *dw = in.grad[2] = &tensor(1, C0, C1, 1).map(O_FILL, DU0); ///> dw
    Tensor *b  = in.grad[1] = &vector(C0).map(O_FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &vector(C0).map(O_FILL, DU0);           ///> db
    
    DU k = DU1 / SQRT(C1);                       /// * default weight
    _mmu->random(*w, UNIFORM, -0.5, 2.0 * k);    /// * randomize w
    printf("bias=%4.2f,  k=%6.3f, w.std=%6.3f\n", bias, k, w->std());
    
    Tensor &out = vector(C0);                    ///> output tensor sizing
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
Model::_ipool(Tensor &in, U16 f) {
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
