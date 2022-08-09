/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
///
/// Convolution and Linear ops
///
__GPU__ void
Model::iconv2d(DU bias, IU c, U16 *opt) {
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
    push(out);                                   /// * stage for next stage
}
__GPU__ void
Model::ilinear(DU bias, U16 n) {
    Tensor &in = *nten; in.grad_fn = DLINEAR;    ///> derivative function

    U16 m = in.H();
    Tensor *w  = in.grad[0] = &tensor(1, n, m, 1);                 ///> w
    Tensor *dw = in.grad[2] = &tensor(1, n, m, 1).map(FILL, DU0);  ///> dw
    Tensor *b  = in.grad[1] = &vector(n).map(FILL, bias);          ///> b
    Tensor *db = in.grad[3] = &vector(n).map(FILL, DU0);           ///> db
    _mmu->random(*w, NORMAL);                    /// * randomize w
    
    Tensor &out = vector(n);                     ///> output tensor sizing
    push(out);                                   /// * stage for next stage
}
__GPU__ void
Model::iflatten() {
    Tensor &in  = *nten;
    Tensor &out = vector(in.size);
    in.grad_fn  = DFLATTEN;
    in.parm     = in.size;
    push(out);
}
///
/// Activation ops
///
__GPU__ void
Model::irelu() {
    Tensor &in  = *nten;
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    in.grad_fn  = DRELU;
    push(out);                    /// * stage for next stage
}
__GPU__ void
Model::itanh() {
}
__GPU__ void
Model::isigmoid() {
}
__GPU__ void
Model::isoftmax() {
    Tensor &in  = *nten;
    Tensor &out = _mmu->copy(in); ///> output tensor sizing
    in.grad_fn  = DSOFTMAX;
    push(out);                    /// * stage for next stage
}
///
/// Pooling and Dropout ops
///
__GPU__ void
Model::imaxpool(U16 f) {
    Tensor &in  = *nten; in.grad_fn = DMAXPOOL;
    in.parm     = f;
    
    U16 m = int((in.H() - f) / f) + 1;
    U16 n = int((in.W() - f) / f) + 1;
    U16 s[4] = { 1, f, f, 1 }; memcpy(in.stride, s, sizeof(s));  // stride
    
    Tensor &out = tensor(1, m, n, in.C());
    push(out);                  /// * stage for next stage
}
__GPU__ void
Model::iavgpool(U16 n) {
}
__GPU__ void
Model::iminpool(U16 n) {
}
__GPU__ void
Model::idropout(U16 f) {
    Tensor &in  = *nten;
    Tensor &out = _mmu->copy(in);
    in.grad_fn  = DDROPOUT;
    in.parm     = f;
    push(out);
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
