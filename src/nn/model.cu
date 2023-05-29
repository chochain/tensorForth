/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network model constructor implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
#define NAME_LIST {                                        \
    "output ", "conv2d ", "linear ", "flatten", "relu   ", \
    "tanh   ", "sigmoid", "softmax", "logsmax", "avgpool", \
    "maxpool", "minpool", "dropout", "upsampl", "bnormal" }
    
__HOST__ const char*                ///< host network layer name 
Model::nname(int i) {
    static const char *name[] = NAME_LIST;
    return name[i];
}
__GPU__ const char*                ///< device network layer name 
Model::d_nname(int i) {
    static const char* name[] = NAME_LIST;
    return name[i];
}
///
/// NN layer factory
///
__GPU__ Model&
Model::add(t4_layer fn, U16 n, DU bias, U16 *opt) {
    Tensor &in = (*this)[-1];
    if (!autograd || in.grad_fn != L_NONE) return *this;
    
    switch(fn) {
    case L_CONV:    _iconv(in, n, bias, opt);   break;
    case L_LINEAR:  _ilinear(in, n, bias);      break;
    case L_FLATTEN: _iflatten(in);              break;
    case L_RELU:
    case L_TANH:
    case L_SIGMOID: _icopy(in);                 break;
    case L_LEAKYRL:
    case L_ELU:     _iactivate(in, bias);       break;
    case L_SOFTMAX:
    case L_LOGSMAX: _isoftmax(in);              break;
    case L_AVGPOOL:
    case L_MAXPOOL:
    case L_MINPOOL: _ipool(in, n);              break;
    case L_DROPOUT: _idropout(in, bias);        break;
    case L_USAMPLE: _iup(in, n, bias);          break;
    case L_BNORMAL: _ibatchnorm(in);            break;
    default: ERROR("Model#add layer %d not supported\n", fn);
    }
    in.grad_fn = fn;

    return *this;
}
///
/// Convolution and Linear ops
///
__GPU__ void
Model::_iconv(Tensor &in, U16 C0, DU bias, U16 *opt) {
    U16 N1 = in.N(), C1 = in.C();                     ///> batch_sz, channels
    U16 Hf = opt[0], Wf = opt[1];                     ///> filter sizing
    U16 p  = (Hf>1&&opt[2]) ? opt[2] : INT((Hf-1)/2); ///> padding
    U16 s  = opt[3], d = opt[4];                      ///> stride, dilation
    U16 H0 = (in.H() - Hf + p*2) / s + 1;             ///> output height
    U16 W0 = (in.W() - Wf + p*2) / s + 1;             ///> output width
    if (Hf != Wf || (Hf != 1 && Hf != 3 && Hf != 5)) {
        ERROR("Model#conv2d f=[%d,%d]? 1x1, 3x3, and 5x5 supported only.\n", Hf, Wf);
        return;
    }
    in.stride[0] = in.stride[1] = s;
    in.parm = INT(bias * 1000.0);
    ///
    /// filter: C1 to C0 fully connected
    /// TODO: filters's 5th dimension is stored in parm field for now
    ///
    Tensor *f  = in.grad[0] = &_t4(C1, Hf, Wf, C0);                  ///> f
    Tensor *df = in.grad[2] = &_t4(C1, Hf, Wf, C0).map(O_FILL, DU0); ///> df
    Tensor *b  = in.grad[1] = &_vec(C0).map(O_FILL, bias);           ///> b
    Tensor *db = in.grad[3] = &_vec(C0).map(O_FILL, DU0);            ///> db

    DU k = DU1 / SQRT(Hf * Wf * C1);             /// * filter default range
    _mmu->random(*f, UNIFORM, -0.5, 2.0 * k);    /// * randomize f [-k ~ k)
    /* dump filter and bias 
    printf("bias=%4.2f,  k=%6.4f, f.std=%6.4f\n", bias, k, f->std());
    for (int i=0; i<f->numel; i++) {
        DU dx = f->data[i];
        printf("%6.3f", dx);
    }
    */
    Tensor &out= _t4(N1, H0, W0, C0);           ///> output tensor
    npush(out);                                 /// * stage for next stage
}
__GPU__ void
Model::_ilinear(Tensor &in, U16 C0, DU bias) {
    U16 N1 = in.N(), C1 = in.HWC();
    Tensor *w  = in.grad[0] = &_t4(1, C0, C1, 1);                   ///> w
    Tensor *dw = in.grad[2] = &_t4(1, C0, C1, 1).map(O_FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &_vec(C0).map(O_FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &_vec(C0).map(O_FILL, DU0);           ///> db
    
    in.parm = INT(bias * 1000.0);                /// * keep for persistence
    
    DU k = DU1 / SQRT(C1);                       /// * default weight
    _mmu->random(*w, UNIFORM, -0.5, 2.0 * k);    /// * randomize w
    TRACE1("bias=%4.2f,  k=%6.3f, w.std=%6.3f\n", bias, k, w->std());
    
    Tensor &out = _t4(N1, C0);                   ///> output tensor sizing
    TRACE1(" out[%d,%d,%d,%d]", out.N(), out.H(), out.W(), out.C());
    npush(out);                                  /// * stage for next stage
}
__GPU__ void
Model::_iflatten(Tensor &in) {
    in.parm = in.HWC();                          /// * keep numel per sample
    TRACE1("flatten parm=%d\n", in.parm);
    Tensor &out = _t4(in.N(), in.parm);          /// * for backprop
    npush(out);
}
///
/// Activation ops
///
__GPU__ void
Model::_icopy(Tensor &in) {
    Tensor &out = _mmu->copy(in);                ///> output tensor sizing
    npush(out);                                  /// * stage for next stage
}

__GPU__ void
Model::_iactivate(Tensor &in, DU alpha) {
    Tensor &out = _mmu->copy(in);
    
    in.parm = INT(1000.0 * alpha);               /// * alpha * 1000
    TRACE1("alpha=%6.3f\n", alpha);
    
    npush(out);
}

__GPU__ void
Model::_isoftmax(Tensor &in) {
    Tensor *sum = in.grad[0] = &_vec(in.N());    ///> for sum per sample
    Tensor &out = _mmu->copy(in);                ///> output tensor sizing
    npush(out);
}
///
/// Pooling, Dropout, and UpSample ops
///
__GPU__ void
Model::_ipool(Tensor &in, U16 f) {
    if (f != 2 && f != 3) {
        ERROR("Model#pooling f=[%d,%d]? 2x2 and 3x3 supported only\n", f, f);
        return;
    }
    in.parm = f;                                 /// * keep kernel size
                                                 /// * used by backprop
    U16 H0 = INT((in.H() - f) / f) + 1;
    U16 W0 = INT((in.W() - f) / f) + 1;
    U16 s[4] = { f, f, 1, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
    Tensor &out = _t4(in.N(), H0, W0, in.C());
    npush(out);                                  /// * stage for next stage
}

__GPU__ void
Model::_idropout(Tensor &in, DU pct) {
    Tensor &out = _mmu->copy(in);
    Tensor *msk = in.grad[0] = &_mmu->copy(in);  ///> dropout mask
    
    in.parm = INT(1000.0 * pct);                 /// * keep pct * 1000
    TRACE1("dropout=%6.3f\n", pct);
    
    npush(out);
}

__GPU__ void
Model::_iup(Tensor &in, U16 f, DU method) {
    if (f != 2 && f != 3) {
        ERROR("Model#upsample f=[%d,%d]? 2x2 and 3x3 supported only\n", f, f);
        return;
    }
    in.parm = (INT(method)<<8) | f;              /// * keep (method<<8) | kernel size
                                                 /// * used by backprop
    U16 H0 = in.H() * f;
    U16 W0 = in.W() * f;
    U16 s[4] = { f, f, 1, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
    Tensor &out = _t4(in.N(), H0, W0, in.C());
    npush(out);                                  /// * stage for next stage
}

__GPU__ void
Model::_ibatchnorm(Tensor &in) {
    const int C1 = in.C();                       /// C0==C1
    Tensor &out = _mmu->copy(in);                /// * retain dimensions

    in.grad[0] = &_mmu->copy(in);                ///> gamma * x_hat
    in.grad[1] = &_vec(C1).map(O_FILL, DU1);     ///> gamma (scale)
    in.grad[2] = &_vec(C1).map(O_FILL, DU0);     ///> beta (shift)
    in.grad[3] = &_vec(C1*2);                    ///> batch mean, stdvar

    in.parm = INT(1000.0 * 0.1);                 ///> EMA momentum
    
    npush(out);
}

#endif  // T4_ENABLE_OBJ
//==========================================================================
