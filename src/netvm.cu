/** -*- c++ -*-
 * @File
 * @brief - Neural Network Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "netvm.h"

#if T4_ENABLE_OBJ
__GPU__ void
NetVM::nnop(t4_layer op) {
    ///
    /// handle tensor ops (proxy)
    ///
    if (TOS1T) {
        switch (op) {
        case L_RELU:    xop1(O_RELU, DU0); break;   ///> (Ta -- Ta Ta')
        case L_TANH:    xop1(O_TANH);      break;   ///> (Ta -- Ta Ta')
        case L_SIGMOID: xop1(O_SIGM);      break;   ///> (Ta -- Ta Ta')
        case L_FLATTEN:
            Tensor &t = TTOS;
            t.reshape(t.numel);            break;   ///> (Ta -- Ta Ta')
        }
        return;
    }
    ///
    /// model layer ops
    ///
    switch (op) {
    case L_CONV2D:   _conv2d(); break;
    case L_LINEAR:
         if (!IS_OBJ(top) && !IS_OBJ(ss[-1])) {
             U16   n    = POPi;                     ///> number of output channels
             DU    bias = POP();                    ///> convolution bias
             NN.add(L_LINEAR, n, bias);             ///> (N b c -- N')
         }
         else ERROR("linear: bias n required!");
         break;
    case L_FLATTEN:
    case L_RELU:
    case L_TANH:
    case L_SIGMOID:
    case L_SOFTMAX: if (MTOS) NN.add(op); break;
    case L_MAXPOOL: 
    case L_AVGPOOL:
    case L_MINPOOL: if (MNOS) { U16 n = POPi; NN.add(op, n); } break;
    case L_DROPOUT: if (MNOS) {
            U16 p = int(100.0 * POP() + 0.5); NN.add(op, p);
        } break;
    default: ERROR("NetVM::nnop(%d) not supported\n", op);
    }
}
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
    const Code prim[] = {                   ///> singleton, build once only
    ///@defgroup Convolution and Linear ops
    ///@{
    CODE("nn.model",                          ///> (n -- N)
         Model &m = mmu.model(POPi);          /// create model with n layers
         PUSH(mmu.mdl2du(m))),
    CODE("conv2d",    nnop(L_CONV2D)),        ///> (N b c [A] -- N')
    CODE("linear",    nnop(L_LINEAR)),        ///> (N n -- N')
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("relu",      nnop(L_RELU)),          ///> (N -- N')
    CODE("tanh",      nnop(L_TANH)),          ///> (N -- N')
    CODE("sigmoid",   nnop(L_SIGMOID)),       ///> (N -- N')
    CODE("softmax",   nnop(L_SOFTMAX)),       ///> (N -- N')
    ///@}
    ///@defgroup Pooling and Dropout ops
    ///@{
    CODE("pool.max",  nnop(L_MAXPOOL)),       ///> (N n -- N')
    CODE("pool.avg",  nnop(L_AVGPOOL)),       ///> (N n -- N')
    CODE("pool.min",  nnop(L_MINPOOL)),       ///> (N n -- N')
    CODE("dropout",   nnop(L_DROPOUT)),       ///> (N p -- N')
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
    CODE("autograd",  if (MNOS) { bool on = POPi; NN.autograd = on; }),
    CODE("forward",
         if (TOS1T && MNOS) {
             Tensor &t = TTOS; POP(); NN.forward(t);
         }),
    CODE("backprop",
         if (TOS1T && MNOS) {
             Tensor &t = TTOS; POP(); NN.backprop(t);
         }),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        if (MNOS) { DU t = POP(); NN.npush(t); }),
    CODE("n@",        if (MNOS) { DU i = POPi;  PUSH(NN[i]); }),
    CODE("network",   if (MTOS) fout << opx(OP_NET, 0, top)),
    ///@}
    };
    const Code over[] = {           /// extended (overload) words
    CODE("flatten",   nnop(L_FLATTEN)),
    CODE("boot",      mmu.clear(FIND("network") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    mmu.status();
};
#endif  // T4_ENABLE_OBJ
//=======================================================================================
