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
NetVM::_conv2d() {
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
    NTOP.iconv2d(bias, c, opt);
}
///
/// Batch ops
///
__GPU__ void
NetVM::nn_for() {
    Tensor &A = mmu.tensor(1, 28, 28, 1);
}
__GPU__ void
NetVM::nn_next() {
    Tensor &A = mmu.tensor(1, 28, 28, 1);
}
///
/// NN model propegation
///
__GPU__ void
NetVM::forward(Model &model, Tensor &input) {
    Tensor &in = model[1];
    if (!in.is_same_shape(input)) {
        ERROR("model#forward dim?\n");
        return;
    }
    Tensor::copy(input, in);
    for (int i = 2; i < (numel - 1); i++) {
        Tensor &out = model[i];
        model.step(in, out);
        in = out;
    }
}
__GPU__ void
NetVM::backprop(Model &model, Tensor &output) {
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
    CODE("nn.model",  DU m = mmu.mdl2du(mmu.model(POPi)); PUSH(m)),
    CODE("autograd",  bool on = POPi; NTOP.autograd = on),
    CODE("conv2d",    _conv2d()),                          ///> (N b c [A] -- N')
    CODE("linear",                                         ///> (N n -- N')
         if (!IS_OBJ(top) && !IS_OBJ(ss[-1])) {
             U16   n    = POPi;          ///> number of output channels
             DU    bias = POP();         ///> convolution bias
             NTOP.ilinear(bias, n);                        ///> (N b c -- N')
         }
         else ERROR("linear: bias n required!")),
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("relu",      NTOP.irelu()),                       ///> (N -- N')
    CODE("tanh",      NTOP.itanh()),                       ///> (N -- N')
    CODE("sigmoid",   NTOP.isigmoid()),                    ///> (N -- N')
    CODE("softmax",   NTOP.isoftmax()),                    ///> (N -- N')
    ///@}
    ///@defgroup Pooling and Dropout ops
    ///@{
    CODE("pool.max",  U16 n = POPi; NTOP.imaxpool(n)),     ///> (N n -- N')
    CODE("pool.avg",  U16 n = POPi; NTOP.iavgpool(n)),     ///> (N n -- N')
    CODE("pool.min",  U16 n = POPi; NTOP.iminpool(n)),     ///> (N n -- N')
    CODE("dropout",                                        ///> (N p -- N')
         DU p = POP();
         NTOP.idropout(int(100.0 * p + 0.5))),
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("loss.nll",  {}),
    CODE("loss.mse",  {}),
    CODE("loss.ce",   {}),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Batch ops
    ///@{
    CODE("nn.for",    {}),
    CODE("nn.next",   {}),
    CODE("forward",   {}),
    CODE("backprop",  {}),
    ///@}
    ///@defgroup Gradiant ops
    ///@{
    CODE("nn.sgd",    {}),
    CODE("nn.adam",   {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        DU t = POP();   NTOP.npush(t)),
    CODE("network",   fout << opx(OP_NET, 0, top)),
    ///@}
    };
    const Code over[] = {           /// extended (overload) words
    CODE("flatten",
         Tensor &t = mmu.du2ten(top);
         if (t.is_tensor()) t.reshape(t.numel);   /// (Ta -- Ta')
         else NTOP.iflatten()),                   /// (N -- N')
    CODE("boot", mmu.clear(FIND("network") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    mmu.status();
};
#endif  // T4_ENABLE_OBJ
//=======================================================================================
