/** -*- c++ -*-
 * @File
 * @brief - Neural Network Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "netvm.h"

#if T4_ENABLE_OBJ
__GPU__ void
NetVM::nnop(t4_layer op) {     /// vtable dispatcher
    ///
    /// handle tensor ops (proxy)
    ///
    if (TOS1T) {
        switch (op) {
        case L_RELU:    xop1(O_RELU, DU0); break; ///> (Ta -- Ta Ta')
        case L_TANH:    xop1(O_TANH);      break; ///> (Ta -- Ta Ta')
        case L_SIGMOID: xop1(O_SIGM);      break; ///> (Ta -- Ta Ta')
        case L_FLATTEN:                           ///> (Ta -- Ta Ta')
            Tensor &t = TTOS;
            t.reshape(t.numel);            break; 
        }
        return;
    }
    ///
    /// model layer ops
    ///
    switch (op) {
    case L_CONV:  _conv(); break;
    case L_LINEAR:
        if (M2V) {                                 ///> param checking
            U16   n    = POPi;                     ///> number of output channels
            DU    bias = POP();                    ///> convolution bias
            MTOS.add(L_LINEAR, n, bias);           ///> (N b c -- N')
        }
        else ERROR("linear: bias n required!");
        break;
    case L_FLATTEN:
    case L_RELU:
    case L_TANH:
    case L_SIGMOID:
    case L_SOFTMAX: if (IS_M(top)) MTOS.add(op); break;
    case L_MAXPOOL: 
    case L_AVGPOOL:
    case L_MINPOOL: if (M1V) { U16 n = POPi; MTOS.add(op, n); } break;
    case L_DROPOUT: if (M1V) {
            U16 p = int(100.0 * POP() + 0.5); MTOS.add(op, p);
        } break;
    default: ERROR("NetVM::nnop(%d) not supported\n", op);
    }
}

__GPU__ void
NetVM::predict(Tensor &I, Tensor &P) {
}
///===================================================================
/// private functions
///
/// Batch ops
///
__GPU__ void
NetVM::nn_for() {}

__GPU__ void
NetVM::nn_next() {}
///
/// Convolution ops
/// @default: 3x3 filter, padding=1, stride=1, dilation=1
///
__GPU__ void
NetVM::_conv() {
    U16 opt[] = { 3, 3, 1, 1, 1 };   ///> default config vector
    if (TOS1T) {                     ///> if optional vector given
        Tensor &v = TTOS;
        if (v.rank == 1) {
            POP();
            for (int i=0; i<5; i++) opt[i] = (U16)v.data[i];
        }
        else { ERROR("vec?"); return; }
    }
    if (!M2V) {
        ERROR("conv2d bias c required!"); return;
    }
    U16 c    = POPi;                 ///> number of output channels
    DU  bias = POP();                ///> convolution bias
    MTOS.add(L_CONV, c, bias, opt);
}
///
/// loss functions
///
__GPU__ void
NetVM::_loss(t4_loss op) {
    if (!TOS1T || !IS_M(ss[-1])) { ERROR("target tensor?\n"); return; }
    Tensor &t = TTOS;
    DU      n = MNOS.loss(op, t);
    printf("NetVM#loss => %.3f", n);
    PUSH(n);
}
///
/// gradiant ops
///
__GPU__ void
NetVM::_sgd() {}

__GPU__ void
NetVM::_adam() {}

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
    CODE("nn.model",                          ///> (n h w c -- N)
         if (ss.idx < 4 ||                    /// * param check
             IS_OBJ(top) || IS_OBJ(ss[-1]) ||
             IS_OBJ(ss[-2]) || IS_OBJ(ss[-3])) {
             ERROR("n h w c?\n"); return;
         }
         U16 c=POPi; U16 w=POPi; U16 h=POPi; U16 n=POPi;
         Model  &m = mmu.model();             /// * create NN model
         Tensor &t = mmu.tensor(n,h,w,c);     /// * create input tensor
         m.npush(t);                          /// * serves as the 1st layer
         PUSH(m)),
    CODE("conv2d",    nnop(L_CONV)),          ///> (N b c [A] -- N')
    CODE("linear",    nnop(L_LINEAR)),        ///> (N b n -- N')
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
    CODE("maxpool",   nnop(L_MAXPOOL)),       ///> (N n -- N')
    CODE("avgpool",   nnop(L_AVGPOOL)),       ///> (N n -- N')
    CODE("minpool",   nnop(L_MINPOOL)),       ///> (N n -- N')
    CODE("dropout",   nnop(L_DROPOUT)),       ///> (N p -- N')
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("loss.nll",  _loss(LOSS_NLL)),       ///> (N T -- N T n)
    CODE("loss.mse",  _loss(LOSS_MSE)),       ///> (N T -- N T n)
    CODE("loss.ce",   _loss(LOSS_CE)),        ///> (N T -- N T n)
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
    CODE("autograd",  if (M1V) { bool on = POPi; MTOS.autograd = on; }),
    CODE("forward", 
        if (TOS1T && IS_M(ss[-1])) {
            Tensor &t = TTOS; POP();
            MTOS.forward(t);
        }
        else ERROR("N set?\n")),
    CODE("backprop",
         if (TOS1T && IS_M(ss[-1])) {
             Tensor &t  = TTOS; POP();
             MTOS.backprop(t);
         }
         else ERROR("N tgt?\n")),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        if (M1V) { DU t = POP(); MTOS.npush(t); }),
    CODE("n@",        if (M1V) { I16 i = POPi; PUSH(mmu.view(MTOS[i])); }),
    CODE("network",   if (IS_M(top)) fout << top),
    CODE("load",
        char *fname = next_idiom();    /// retrieve file name
        fout << opx(OP_LOAD, 0, top) << fname;
        state = VM_WAIT),
    ///@}
    };
    const Code over[] = {              ///< extended (overload) words
    CODE("flatten",   nnop(L_FLATTEN)),
    CODE("boot",      mmu.clear(FIND("network") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    
    VLOG1("NetVM::init ok\n");
};
#endif  // T4_ENABLE_OBJ
//===========================================================================
