/** -*- c++ -*-
 * @File
 * @brief - Neural Network Model implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "model.h"

#if T4_ENABLE_OBJ
///
/// Convolution ops
///
__GPU__ void
Model::iconv2d(DU bias, IU c, U16 *opt) {
    Tensor &in = *nten; in.grad_fn = DCONV2D;    ///> derivative function

    U16 m = opt[0], n = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : floor((m-1)/2);    ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation

    Tensor *w  = in.grad[0] = &tensor(1, m, n, c);                 ///> w
    Tensor *b  = in.grad[1] = &tensor(1, 1, 1, c).map(FILL, bias); ///> b
    Tensor *dw = in.grad[2] = &tensor(1, m, n, c).map(FILL, DU0);  ///> dw
    Tensor *db = in.grad[3] = &tensor(1, 1, 1, c).map(FILL, DU0);  ///> db
    mmu.random(*w, NORMAL);                      /// * randomize w
    
    Tensor &out = tensor(                        ///> output tensor sizing
        1,
        in.H() + 2 * (p - floor(m/2)),
        in.W() + 2 * (p - floor(n/2)),
        c);
    push(out);                                   /// * stage for next stage
}
///
/// Pooling ops
///
__GPU__ void
Model::imaxpool(U16 n) {
    Tensor &in  = *nten; in.grad_fn = DMAXPOOL;
    U16 m = floor((in.H() - n) / n) + 1;
    U16 n = floor((in.W() - n) / n) + 1;
    Tensor &out = tensor(1, m, n, in.C());
    in.parm     = n;
    push(out);                  /// * stage for next stage
}
__GPU__ void
Model::imeanpool(U16 n) {
}
__GPU__ void
Model::iavgpool(U16 n) {
}
__GPU__ void
Model::iminpool(U16 n) {
}
///
/// Activation ops
///
__GPU__ void
Model::irelu() {
    Tensor &in  = *nten; in.grad_fn = DRELU;
    Tensor &out = mmu.copy(in); ///> output tensor sizing
    push(out);                  /// * stage for next stage
}
__GPU__ void
Model::tanh() {
}
__GPU__ void
Model::sigmoid() {
}
__GPU__ void
Model::softmax() {
}
///
/// Pooling ops
///
__GPU__ void
Model::linear(U16 c) {
    Tensor &in = *nten; in.grad_fn = DLINEAR;    ///> derivative function

    U16 m = in.H(), n = in.W();                  ///> filter sizing
    Tensor *w  = in.grad[0] = &tensor(1, m, n, c);                 ///> w
    Tensor *b  = in.grad[1] = &tensor(1, 1, 1, c).map(FILL, bias); ///> b
    Tensor *dw = in.grad[2] = &tensor(1, m, n, c).map(FILL, DU0);  ///> dw
    Tensor *db = in.grad[3] = &tensor(1, 1, 1, c).map(FILL, DU0);  ///> db
    mmu.random(*w, NORMAL);                      /// * randomize w
    
    Tensor &out = tensor(1, m, n, c);            ///> output tensor sizing
    push(out);                                   /// * stage for next stage
}
///
/// Pooling ops
///
__GPU__ void
Model::dropout(U16 p) {
    Tensor &in = *nten; in.grad_fn = DDROPOUT;   ///> derivative function
}
#endif  // T4_ENABLE_OBJ
//=======================================================================================
