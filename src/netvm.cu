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
        if (MN2D) {                                ///> param checking
            U16   n    = POPi;                     ///> number of output channels
            DU    bias = POP();                    ///> convolution bias
            NN0.add(L_LINEAR, n, bias);            ///> (N b c -- N')
        }
        else ERROR("linear: bias n required!");
        break;
    case L_FLATTEN:
    case L_RELU:
    case L_TANH:
    case L_SIGMOID:
    case L_SOFTMAX: if (MTOS) NN0.add(op); break;
    case L_MAXPOOL: 
    case L_AVGPOOL:
    case L_MINPOOL: if (MNOS) { U16 n = POPi; NN0.add(op, n); } break;
    case L_DROPOUT: if (MNOS) {
            U16 p = int(100.0 * POP() + 0.5); NN0.add(op, p);
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
    if (!MN2D) {
        ERROR("conv2d bias c required!"); return;
    }
    U16 c    = POPi;                 ///> number of output channels
    DU  bias = POP();                ///> convolution bias
    NN0.add(L_CONV, c, bias, opt);
}
///
/// loss functions
///
__GPU__ void
NetVM::_loss(t4_loss op, Tensor &A, Tensor &B) {
    if (!A.is_same_shape(B)) { ERROR("same size?\n"); return; }
    U16 SZ = A.numel;
    DU  *da = A.data, *db = B.data;
    DU  rst = DU0;
    switch (op) {
    case LOSS_MSE: {
        for (int i=0; i < SZ; i++) {
            DU v = *da++ - *db++;
            rst += v * v;
        }
        printf("NetVM#mse sum=%.3f, N=%d => %.3f",
               rst, A.N(), rst / (2.0*A.N()));
        rst /= (2.0 * A.N());
    } break;
    case LOSS_NLL: break;
    case LOSS_CE:  break;
    default: ERROR("loss funtion %d not supported\n", op);
    }
    PUSH(rst);
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
    CODE("loss.nll",  if (TOS2T) _loss(LOSS_NLL, TTOS, TNOS)),
    CODE("loss.mse",  if (TOS2T) _loss(LOSS_MSE, TTOS, TNOS)),
    CODE("loss.ce",   if (TOS2T) _loss(LOSS_CE,  TTOS, TNOS)),
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
    CODE("autograd",  if (MNOS) { bool on = POPi; NN0.autograd = on; }),
    CODE("forward", 
        if (TOS1T && IS_M(ss[-1])) {
            Tensor &t = TTOS; POP();
            NN0.forward(t);
        }
        else ERROR("N set?\n")),
    CODE("backprop",
         if (TOS1T && IS_M(ss[-1])) {
             Tensor &t  = TTOS; POP();
             NN0.backprop(t);
         }
         else ERROR("N tgt?\n")),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        if (MNOS) { DU t = POP(); NN0.npush(t); }),
    CODE("n@",        if (MNOS) { I16 i = POPi; PUSH(mmu.view(NN0[i]));
        }),
    CODE("network",   if (MTOS) fout << top),
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
//===========================================================================
