/** -*- c++ -*-
 * @File
 * @brief - Neural Network Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "netvm.h"

#if T4_ENABLE_OBJ
///===================================================================
/// static loss functions
///
__GPU__ void
NetVM::loss_nll(Tensor &A, Tensor &B, Tensor &C) {
}
__GPU__ void
NetVM::loss_mse(Tensor &A, Tensor &B, Tensor &C) {
}
__GPU__ void
NetVM::loss_ce(Tensor &A, Tensor &B, Tensor &C) {
}
__GPU__ void
NetVM::predict(Tensor &A, Tensor &B, Tensor &C) {
}
__GPU__ void
NetVM::dconv2d(Tensor &A, Tensor &B) {
}
__GPU__ void
NetVM::drelu(Tensor &A, Tensor &B) {
}
__GPU__ void
dmaxpool(Tensor &A, Tensor &B) {
}
__GPU__ void
dreshape(Tensor &A, Tensor &B) {
}
__GPU__ void
dlinear(Tensor &A, Tensor &B) {
}
///
/// Convolution ops
///
__GPU__ void
NetVM::conv2d(DU bias, U16 c) {
    const U16 opt[] = { 3, 3, 1, 1, 1 };
    conv2d(bias, c, (U16*)opt);
}
__GPU__ void
NetVM::conv2d(DU bias, U16 c, U16 *opt) {
    if (!ttop) return;
    Tensor &t = *ttop;                         ///> TOS tensor
    if (!t.grad_fn) initgrad(t, bias, c, opt); ///> create tensors if not yet
    ///
    /// apply convolution filter
    ///
}
///
/// Activation ops
///
__GPU__ void
NetVM::relu() {
}
__GPU__ void
NetVM::tanh() {
}
__GPU__ void
NetVM::sigmoid() {
}
__GPU__ void
NetVM::softmax() {
}
///
/// Pooling ops
///
__GPU__ void
NetVM::meanpool(U16 n) {
}
__GPU__ void
NetVM::avgpool(U16 n) {
}
__GPU__ void
NetVM::maxpool(U16 n) {
}
__GPU__ void
NetVM::minpool(U16 n) {
}
///
/// Pooling ops
///
__GPU__ void
NetVM::linear(U16 n) {
}
///
/// Pooling ops
///
__GPU__ void
NetVM::dropout(U16 p) {
}
///
/// Back Propegation ops
///
__GPU__ void
NetVM::initgrad(Tensor &A, DU bias, U16 c, U16 *opt) {
    /// create autograd tensors
    ///
    A.grad_fn = &dconv2d;                        ///> derivative function

    U16 m = opt[0], n = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : floor((m-1)/2);    ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation
    
    Tensor *w  = A.grad[0] = &mmu.tensor(1, m, n, c);                   ///> w
    mmu.random(*w, NORMAL);
    Tensor *b  = A.grad[1] = &mmu.tensor(1, 1, 1, c).map(FILL, bias); ///> b
    Tensor *dw = A.grad[2] = &mmu.tensor(1, m, n, c).map(FILL, DU0);  ///> dw
    Tensor *db = A.grad[3] = &mmu.tensor(1, 1, 1, c).map(FILL, DU0);  ///> db
}
__GPU__ void
NetVM::autograd(bool on) {
}
__GPU__ void
NetVM::batch_for() {
}
__GPU__ void
NetVM::backprop() {
}
__GPU__ void
NetVM::sgd() {
}
__GPU__ void
NetVM::adam() {
}

///===================================================================
/// class methods
///
/// Neural Network specific dictionary constructor
///
__GPU__ void
NetVM::init() {
    const Code prim[] = {       /// singleton, build once only
    ///@defgroup Tensor creation ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("conv2d",    {}),      ///< allocate a vector
    CODE("conv2d",    {}),      ///< allocate a matrix
    CODE("pad2d",     {}),      ///< allocate a NHWC tensor
    CODE("padr2d",    {}),      ///< create a vector with literals
    ///@}
    ///@defgroup Tensor shape ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("relu",      {}),      ///< create a matrix with literals
    CODE("tanh",      {}),
    CODE("sigmoid",   {}),
    CODE("softmax",   {}),
    ///@}
    ///@defgroup Tensor shape ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("meanpool",  {}),
    CODE("avgpool",   {}),
    CODE("maxpool",   {}),
    CODE("minpool",   {}),
    ///@}
    ///@defgroup Tensor fill ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("linear",    {}),
    CODE("loss_nll",  {}),
    CODE("loss_mse",  {}),
    CODE("loss_ce",   {}),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Tensor fill ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("batch_for", {}),
    CODE("batch_next",{}),
    CODE("sgd",       {}),
    CODE("adam",      {}),
    CODE("autograd",  {})
    ///@}
    };
    const Code over[] = {          /// extended (overload) words
    CODE("boot", mmu.clear(FIND("autograd") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    mmu.status();
};
#endif  // T4_ENABLE_OBJ
//=======================================================================================
