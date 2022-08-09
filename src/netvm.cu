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
///
/// Convolution ops
///
__GPU__ void
NetVM::conv2d() {
    U16 opt[] = { 3, 3, 1, 1, 1 };   ///> default 3x3 filter, padding=1, stride=1, dilation=1
    if (IS_TEN(top)) {
        Tensor &v = mmu.du2ten(top);
        if (v.rank == 1) {
            POP();
            for (int i=0; i<5; i++) opt[i] = (U16)v.data[i];
        }
        else { ERROR("vec?"); return; }
    }
    if (IS_OBJ(top) || IS_OBJ(ss[-1])) {
        ERROR("conv2d bias c required!"); return;
    }
    
    U16 c      = POPi;                      ///> number of output channels
    DU  bias   = POP();                     ///> convolution bias

    if (wet()) model.iconv2d(bias, c, opt); /// create autograd tensors if needed
    ///
    /// perform 2D convolution
    ///
}
///
/// Pooling ops
///
__GPU__ void
NetVM::maxpool() {
    if (IS_OBJ(top)) { ERROR("#maxpool n required!"); return; }
    U16 n = POPi;
    if (wet()) model.imaxpool(n);
    ///
    /// perform maxpool
    ///
}
///
/// Activation ops
///
__GPU__ void
NetVM::relu() {
    if (wet()) model.irelu();
    ///
    /// perform ReLU
    ///
}
///
/// Linear ops
///
__GPU__ void
NetVM::linear() {
    if (IS_OBJ(top)) { ERROR("#linear n required!"); return; }
    U16 n = POPi;
    if (wet()) model.ilinear(n);
    ///
    /// perform linear transformation
    ///
}
///
/// Back Propegation ops
///
__GPU__ void
NetVM::autograd(bool on) {
    f_auto = on;
}
__GPU__ void
NetVM::for_batch() {
    Tensor &A = mmu.tensor(1, 28, 28, 1);
    model.push(A);
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
    ///@defgroup Convolution ops
    ///@{
    CODE("conv2d",    conv2d()),     ///> (Ta b c [A] -- Ta')
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("relu",      relu()),       ///> (Ta -- Ta')
    CODE("tanh",      {}),
    CODE("sigmoid",   {}),
    CODE("softmax",   {}),
    ///@}
    ///@defgroup Pooling ops
    ///@{
    CODE("maxpool",   maxpool()),    ///> (Ta n -- Ta')
    CODE("meanpool",  {}),
    CODE("avgpool",   {}),
    CODE("minpool",   {}),
    ///@}
    ///@defgroup Linear ops
    ///@{
    CODE("linear",    linear()),     ///> (Ta n -- Ta')
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("loss_nll",  {}),
    CODE("loss_mse",  {}),
    CODE("loss_ce",   {}),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Tensor fill ops
    ///@{
    CODE("batch_for", {}),
    CODE("batch_next",{}),
    CODE("sgd",       {}),
    CODE("adam",      {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE("network",   fout << opx(OP_NET, model.idx, model.data[0])),
    CODE(">n",        model.push(top); POP()),
    CODE("n>",        DU t = model.pop(); PUSH(t)),
    CODE("autograd",  autograd(POPi)),
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
