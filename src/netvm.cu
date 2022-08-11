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
/// Convolution and Linear ops
///
__GPU__ void
NetVM::conv2d() {
    U16 opt[] = { 3, 3, 1, 1, 1 };   ///> default 3x3 filter, padding=1, stride=1, dilation=1
    if (IS_OBJ(top)) {
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
    U16   c    = POPi;                        ///> number of output channels
    DU    bias = POP();                       ///> convolution bias
    Model &md  = NTOP.iconv2d(bias, c, opt);
    ///
    /// perform 2D convolution
    ///
}
__GPU__ void
NetVM::linear() {
    if (IS_OBJ(top) || IS_OBJ(ss[-1])) {
        ERROR("linear bias n required!"); return;
    }
    U16   n    = POPi;                        ///> number of output channels
    DU    bias = POP();                       ///> convolution bias
    Model &md  = NTOP.ilinear(bias, n);
    ///
    /// perform linear transformation
    ///
}
__GPU__ void
NetVM::flatten() {
    Model &md = NTOP.iflatten();
    ///
    /// flatten input tensor
}
///
/// Activation ops
///
__GPU__ void
NetVM::relu() {
    Model &md = NTOP.irelu();
    ///
    /// perform ReLU
    ///
}
__GPU__ void
NetVM::softmax() {
    Model &md = NTOP.isoftmax();
    ///
    /// perform ReLU
    ///
}
///
/// Pooling and Dropout ops
///
__GPU__ void
NetVM::maxpool() {
    U16 n = POPi;
    Model &md = NTOP.imaxpool(n);
    ///
    /// perform maxpool
    ///
}
__GPU__ void
NetVM::dropout() {
    DU p = POP();
    Model &md = NTOP.idropout(int(100.0 * p + 0.5));
}
///
/// Back Propegation ops
///
__GPU__ void
NetVM::for_batch() {
    Tensor &A = mmu.tensor(1, 28, 28, 1);
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
    ///@defgroup Convolution and Linear ops
    ///@{
    CODE("model",     DU m = mmu.mdl2du(mmu.model(POPi)); PUSH(m)),
    CODE("conv2d",    conv2d()),     ///> (N b c [A] -- N')
    CODE("linear",    linear()),     ///> (N n -- N')
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("relu",      relu()),       ///> (N -- N')
    CODE("tanh",      {}),
    CODE("sigmoid",   {}),
    CODE("softmax",   softmax()),
    ///@}
    ///@defgroup Pooling and Dropout ops
    ///@{
    CODE("maxpool",   maxpool()),    ///> (N n -- N')
    CODE("avgpool",   {}),
    CODE("minpool",   {}),
    CODE("dropout",   dropout()),    ///> (N p -- N')
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
    CODE("network",  fout << opx(OP_NET, 0, top)),
    CODE(">n",       DU t = POP();   NTOP.npush(t)),
    CODE("autograd", bool on = POPi; NTOP.autograd = on)
    ///@}
    };
    const Code over[] = {          /// extended (overload) words
    CODE("flatten",                /// (Ta -- Ta')
         if (mmu.is_tensor(top)) {
             Tensor &t = mmu.du2ten(top);
             t.reshape(t.size);
         }
         else flatten()),          /// (N -- N')
    CODE("boot", mmu.clear(FIND("autograd") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    mmu.status();
};
#endif  // T4_ENABLE_OBJ
//=======================================================================================
