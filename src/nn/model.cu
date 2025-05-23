/** -*- c++ -*-
 * @file
 * @brief Model class - Neural Network model constructor implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "model.h"
///
/// @name name string short hands
/// @{
__HOST__ const char*                ///< host network layer name 
Model::nname(int i) {
    static const char *name[] = { LAYER_OP };
    return name[i];
}
__GPU__ const char*                ///< device network layer name 
Model::d_nname(int i) {
    static const char* name[] = { LAYER_OP };
    return name[i];
}
/// @}
/// @{
/// @name constructor (indirect)
__GPU__  void
Model::init(MMU *mmu, Tensor &store, int &trace) {
    T4Base::init(0, T4_MODEL, 0);           /// * T4Base attributes
    _mmu     = mmu;                         /// * cached memory controller
    _store   = &store;
    _trace   = &trace;
    data     = store.data;                  /// * cached entries
    train    = 1;
    epoch    = 0;
    max_norm = DU0;                         /// * > DU0 to clip norm (more stable)
    npush(store);                           /// * model.data[0] = store
}
/// @}
/// @name layer access methods
/// @{
__BOTH__ Tensor&
Model::operator[](S64 i) {
    /// * model.data[0] = store
    /// so 1st layer starts from model.data[1]
    return (Tensor&)_mmu->du2obj(data[(i < 0L) ? numel + i : i]);
}
__BOTH__ int
Model::slots() { return _store->numel; }

__GPU__ Model&
Model::npush(DU v) {
    data[numel++] = v;
    U32 tsz = _store->numel;                ///< current allocated for layers
    if (tsz <= numel) {                     /// * resize if too many layers
        _mmu->resize(*_store, tsz + T4_NET_SZ);
        data = _store->data;                /// * reset storage cached pointer
    }
    return *this;
}
__GPU__ Model& Model::npush(Tensor &t) { return npush(_mmu->obj2du(t)); }
__GPU__ DU     Model::npop()           { return data[--numel];  }
__GPU__ int    Model::batch_size()     { return (*this)[1].N(); }
/// @{
/// @name Tensor constructors and randomizer
/// @{
__GPU__ Tensor&
Model::COPY(Tensor &t)   { return _mmu->copy(t); }
__GPU__ void
Model::FREE(Tensor &t)   { _mmu->free(t); }
__GPU__ Tensor&
Model::VEC(U64 sz)       { return _mmu->tensor(sz); }
__GPU__ Tensor&
Model::T4(U32 n, U32 h)  { return _mmu->tensor(n, h, 1, 1); }
__GPU__ Tensor&
Model::T4(U32 n, U32 h, U32 w, U32 c) { return _mmu->tensor(n, h, w, c); }
__GPU__ void
Model::RAND(Tensor &t, DU scale) {           ///< short hand to System::rand
    NN_DB("sys#rand(T%d) numel=%ld bias=%.2f, scale=%.2f\n",
          t.rank, t.numel, -0.5, scale*2.0);
    System::rand(t.data, t.numel, UNIFORM, -0.5, scale * 2.0); /// * range=>[-scale, scale)
}
/// @}
/// @name NN layer factory
/// @{
__GPU__ Model&
Model::add(t4_layer fn, U32 n, DU bias, U16 *opt) {
    Tensor &in = (*this)[-1];
    if (in.grad_fn != L_NONE) return *this;    /// * tensor already setup

    NLOG("  Model::add %s n=%d bias=%g {\n", d_nname(fn), n, bias);

    for (int i=0; i<4; i++) in.grad[i] = in.mtum[i] = NULL;
    switch(fn) {
    case L_CONV:    _iconv(in, n, bias, opt);   break;
    case L_LINEAR:  _ilinear(in, n, bias);      break;
    case L_FLATTEN: _iflatten(in);              break;
    case L_RELU:
    case L_TANH:
    case L_SIGMOID:
    case L_SELU:
    case L_LEAKYRL:
    case L_ELU:
    case L_DROPOUT: _iactivate(in, bias);       break;
    case L_SOFTMAX:
    case L_LOGSMAX: _isoftmax(in);              break;
    case L_AVGPOOL:
    case L_MAXPOOL:
    case L_MINPOOL: _ipool(in, (U16)n);         break;
    case L_BATCHNM: _ibatchnorm(in, bias);      break;
    case L_USAMPLE: _iup(in, n, bias);          break;
    default: ERROR("Model#add layer %d not supported\n", fn);
    }
    in.grad_fn = fn;                           /// * set layer function name
    Tensor &out = (*this)[-1];                 /// * output tensor
    NLOG("  } Model::add %s => out[%d,%d,%d,%d]\n",
          d_nname(fn), out.N(), out.H(), out.W(), out.C());
    
    return *this;
}
/// @}
/// @name Convolution and Linear ops
/// @{
__GPU__ void
Model::_iconv(Tensor &in, U32 C0, DU bias, U16 *opt) {
    U32 N1 = in.N(), C1 = in.C();                     ///> batch_sz, channels
    U16 Hf = opt[0], Wf = opt[1];                     ///> filter sizing
    U16 p  = (Hf>1&&opt[2]) ? opt[2] : (Hf-1)/2;      ///> padding
    U16 s  = opt[3], d = opt[4];                      ///> stride, dilation
    U16 H0 = (in.H() - Hf + p*2) / s + 1;             ///> output height
    U16 W0 = (in.W() - Wf + p*2) / s + 1;             ///> output width
    
    NN_DB("    model#iconv %dx%d bias=%4.2f {\n", Hf, Wf, bias);
    if (Hf != Wf || (Hf != 1 && Hf != 3 && Hf != 5)) {
        ERROR("model#iconv f=[%d,%d]? 1x1, 3x3, and 5x5 supported only.\n", Hf, Wf);
        return;
    }
    in.stride[0] = in.stride[1] = s;
    in.xparm = bias;
    ///
    /// filter: C1 to C0 fully connected
    /// TODO: filters's 5th dimension is stored in parm field for now
    ///
    Tensor *f  = in.grad[0] = &T4(C1, Hf, Wf, C0);                 ///> f
    Tensor *df = in.grad[2] = &T4(C1, Hf, Wf, C0).map(FILL, DU0);  ///> df
    Tensor *b  = in.grad[1] = &VEC(C0);                            ///> b
    Tensor *db = in.grad[3] = &VEC(C0).map(FILL, DU0);             ///> db

    DU k = SQRT(RCP(Hf * Wf * C1));              /// * filter default range
    RAND(*f, k);                                 /// * randomize f [-k, k)
    RAND(*b, bias);                              /// * randomize b [-bias, bias)
    
#if MM_DEBUG    
    f->map(FILL, 0.5);                           /// * debug
    b->map(FILL, -0.5);
    
    NN_DB("    f[%d,%d,%d,%d]=", C1, Hf, Hf, C0);
    for (U64 i=0; i<f->numel; i++) NN_DB("%6.3f", f->data[i]);
    NN_DB("\n");
#endif // MM_DEBUG
    
    Tensor &out= T4(N1, H0, W0, C0);             ///> output tensor
    npush(out);                                  /// * stage for next stage
    NN_DB("    } model#iconv => k=%6.3f, f.std=%6.3f\n",  k, f->std());
}
__GPU__ void
Model::_ilinear(Tensor &in, U32 E0, DU bias) {
    NN_DB("    model#ilinear bias=%4.2f {\n", bias);
    U32 N1 = in.N();
    U64 E1 = in.HWC();
    Tensor *w  = in.grad[0] = &T4(1, E0, E1, 1);                  ///> w
    Tensor *dw = in.grad[2] = &T4(1, E0, E1, 1).map(FILL, DU0);   ///> dw
    Tensor *b  = in.grad[1] = &VEC(E0);                           ///> b
    Tensor *db = in.grad[3] = &VEC(E0).map(FILL, DU0);            ///> db
    
    in.xparm = bias;                              /// * keep for persistence
    
    DU k = SQRT(2.0 / (E0+E1));                   /// * default weight
    RAND(*w, k);                                  /// * randomize w [-k, k)
    RAND(*b, bias);                               /// * randomize b [-bias, bias)
    
#if MM_DEBUG    
    w->map(FILL, 0.5); w->data[32] = 1.0;
    b->map(FILL, 0.0);
    
    NN_DB("    w[1,%d,%ld,1]", E0, E1);
    for (U32 e0=0; e0<E0; e0++) {
        NN_DB("\ne0=%d ", e0);
        for (U64 e1=0; e1<E1; e1++) {
            NN_DB("%5.2f", w->data[E1*e0 + e1]);
        }
    }
    NN_DB("\n");
#endif // MM_DEBUG    
    
    Tensor &out = T4(N1, E0);                    ///> output tensor sizing
    npush(out);                                  /// * stage for next stage
    NN_DB("    } model#ilinear => k=%6.3f, w.std=%6.3f\n", k, w->std());
}
__GPU__ void
Model::_iflatten(Tensor &in) {
    NN_DB("    model#iflatten {\n");
    Tensor &out = T4(in.N(), (U32)in.HWC());     /// * for backprop
    npush(out);
    NN_DB("    } model#iflatten\n");
}
/// @}
/// @name Activation ops
/// @{
__GPU__ void
Model::_isoftmax(Tensor &in) {
	NN_DB("    model#isoftmax {\n");
    Tensor &out = COPY(in);                      ///> output tensor sizing
    in.grad[0] = &T4(1,in.H(),in.W(), in.C());   ///> activation mask
    
    npush(out);                                  /// * stage for next stage
	NN_DB("    } model#isoftmax\n");
}

__GPU__ void
Model::_iactivate(Tensor &in, DU alpha) {
    NN_DB("    model#iactivate alpha=%6.3f {\n", alpha);
    Tensor &out = COPY(in);
    Tensor *msk = in.grad[0] = &COPY(in);        ///> activation mask

    in.xparm = alpha;                            /// * keep bias
    
    npush(out);
    NN_DB("    } model#iactivate\n");
}
/// @}
/// @name Pooling, Dropout, and UpSample ops
/// @{
__GPU__ void
Model::_ipool(Tensor &in, U16 f) {
    NN_DB("    model#ipool %dx%d {\n", f, f);
    if (f != 2 && f != 3) {
        ERROR("pooling f=%dx%d? 2x2 and 3x3 supported only\n", f, f);
        return;
    }
    in.iparm = f;                                /// * keep kernel size
                                                 /// * used by backprop
    U32 H0 = (in.H() - f) / f + 1;
    U32 W0 = (in.W() - f) / f + 1;
    U16 s[4] = { f, f, 1, 1 }; memcpy(in.stride, s, sizeof(s));  /// stride
    
    Tensor &out = T4(in.N(), H0, W0, in.C());
    npush(out);                                  /// * stage for next stage
    NN_DB("    } model#ipool\n");
}

__GPU__ void
Model::_ibatchnorm(Tensor &in, DU m) {
    NN_DB("    model#ibatchnorm m=%5.3f {\n", m);
    const int C = in.C();                        /// C0==C1
    in.grad[0] = &VEC(C*2).map(FILL, DU0);       ///> weight/gamma, bias/beta
    in.grad[1] = &VEC(C*2);                      ///> tmp storage
    in.grad[2] = &VEC(C*2).map(FILL, DU0);       ///> d_gamma, d_beta
    in.grad[3] = &COPY(in);                      ///> x_hat (same as in)

    for (int c=0; c < C; c++) {                  /// * default gamma=1.0, beta=0.0
        in.grad[0]->data[c] = DU1;
    }
    in.xparm = m;                                ///> default EMA momentum = 0.1
    
    Tensor &out = COPY(in);                      /// * retain dimensions
    npush(out);
    NN_DB("    } model#ibatchnorm\n");
}

__GPU__ void
Model::_iup(Tensor &in, U16 f, DU method) {
    NN_DB("    model#iup upsample %dx%d {\n", f, f);
    if (f != 2 && f != 3) {
        ERROR("model#upsample f=%dx%d? only 2x2 and 3x3 supported\n", f, f);
        return;
    }
    in.iparm = (INT(D2I(method))<<8) | f;        /// * keep (method<<8) | kernel size
                                                 /// * used by backprop
    U32 H0 = in.H() * f;
    U32 W0 = in.W() * f;
    U16 s[4] = { f, f, 1, 1 }; memcpy(in.stride, s, sizeof(s));  ///< stride
    
    Tensor &out = T4(in.N(), H0, W0, in.C());
    npush(out);                                  /// * stage for next stage
    NN_DB("    } model#iup %dx%d {\n", f, f);
}
/// @}
#endif  // (T4_DO_OBJ && T4_DO_NN)
//==========================================================================
