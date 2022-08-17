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
        Tensor &v = TTOS;
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
    NN.add(L_CONV2D, c, bias, opt);
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
    CODE("nn.conv2d", _conv2d()),                          ///> (N b c [A] -- N')
    CODE("nn.linear",                                      ///> (N n -- N')
         if (!IS_OBJ(top) && !IS_OBJ(ss[-1])) {
             U16   n    = POPi;          ///> number of output channels
             DU    bias = POP();         ///> convolution bias
             NN.add(L_LINEAR, n, bias);                    ///> (N b c -- N')
         }
         else ERROR("linear: bias n required!")),
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("nn.relu",   NN.add(L_RELU)),                     ///> (N -- N')
    CODE("nn.tanh",   NN.add(L_TANH)),                     ///> (N -- N')
    CODE("nn.sigmoid",NN.add(L_SIGMOID)),                  ///> (N -- N')
    CODE("nn.softmax",NN.add(L_SOFTMAX)),                  ///> (N -- N')
    ///@}
    ///@defgroup Pooling and Dropout ops
    ///@{
    CODE("pool.max",  U16 n = POPi; NN.add(L_MAXPOOL, n)), ///> (N n -- N')
    CODE("pool.avg",  U16 n = POPi; NN.add(L_AVGPOOL, n)), ///> (N n -- N')
    CODE("pool.min",  U16 n = POPi; NN.add(L_MINPOOL, n)), ///> (N n -- N')
    CODE("nn.dropout",                                     ///> (N p -- N')
         DU p = POP();
         NN.add(L_DROPOUT, int(100.0 * p + 0.5))),
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("loss.nll",  {}),
    CODE("loss.mse",  {}),
    CODE("loss.ce",   {}),
    ///@}
    ///@defgroup Gradiant ops
    ///@{
    CODE("nn.sgd",    {}),
    CODE("nn.adam",   {}),
    ///@}
    ///@defgroup Batch ops
    ///@{
    CODE("nn.for",    {}),
    CODE("nn.next",   {}),
    CODE("autograd",  bool on = POPi; NN.autograd = on),
    CODE("forward",
         if (!IS_OBJ(top) || !IS_OBJ(ss[-1])) return;
         Tensor &t = mmu.du2ten(POP()); NN.forward(t)),
    CODE("backprop",
         if (!IS_OBJ(top) || !IS_OBJ(ss[-1])) return;
         Tensor &t = mmu.du2ten(POP()); NN.backprop(t)),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        DU t = POP(); NN.npush(t)),
    CODE("n@",        DU i = POPi; PUSH(NN[i])),
    CODE("network",   fout << opx(OP_NET, 0, top)),
    ///@}
    };
    const Code over[] = {           /// extended (overload) words
    CODE("flatten",
         Tensor &t = TTOS;
         if (t.is_tensor()) t.reshape(t.numel);   /// (Ta -- Ta')
         else NN.add(L_FLATTEN)),                 /// (N -- N')
    CODE("boot", mmu.clear(FIND("network") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    mmu.status();
};
#endif  // T4_ENABLE_OBJ
//=======================================================================================
