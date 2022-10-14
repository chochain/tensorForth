/** -*- c++ -*-
 * @file
 * @brief NetVM class - extend TensorVM class, Neural Network Vritual Machine implementation
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
    case L_SOFTMAX:
    case L_LOGSMAX: if (IS_M(top)) MTOS.add(op); break;
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
/// private methods
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
            for (int i=0; i<5; i++) opt[i] = (U16)v.data[i];
            POP(); mmu.free(v);
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
    if (TOS1T && IS_M(ss[-1])) {
        Tensor &t = TTOS; POP();
        DU     n  = MTOS.loss(op, t);
        mmu.free(t);                 /// * pop off t
        PUSH(n);                     /// * loss on TOS
        printf("NetVM#loss => %.3f", n);
    }
    else if (IS_M(top)) PUSH(MTOS.loss(op));
    else ERROR("model?\n");
}

__GPU__ int
NetVM::_fetch() {
    
    return 1;
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
    CODE("batchsize", if (IS_M(top)) PUSH(MTOS.batch_size())),
    CODE("conv2d",    nnop(L_CONV)),          ///> (N b c [A] -- N')
    CODE("linear",    nnop(L_LINEAR)),        ///> (N b n -- N')
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("relu",      nnop(L_RELU)),          ///> (N -- N')
    CODE("tanh",      nnop(L_TANH)),          ///> (N -- N')
    CODE("sigmoid",   nnop(L_SIGMOID)),       ///> (N -- N')
    CODE("softmax",   nnop(L_SOFTMAX)),       ///> (N -- N')
    CODE("logsoftmax",nnop(L_LOGSMAX)),       ///> (N -- N')
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
    CODE("nn.sgd",
         if (!M2V) ERROR("rate mtum nn.sgd?\n");
         else {
             DU m = POP(); DU lr = POP();
             MTOS.sgd(lr, m);
         }),
    CODE("nn.adam",
         if (!M2V) ERROR("rate beta nn.adam?\n");
         else {
             DU b0 = POP(); DU lr = POP();
             DU b1 = M1V ? POP() : DU1 - POW(DU1 - b0, 3);
             MTOS.adam(lr, b0, b1);
         }),
    ///@}
    ///@defgroup Batch ops
    ///@{
    CODE("nn.onehot", if (IS_M(top)) PUSH(MTOS.onehot())),
    CODE("autograd",  if (M1V) { bool on = POPi; MTOS.autograd = on; }),
    CODE("forward",
         if (TOS1D && IS_M(ss[-1])) {           /// dataset on TOS
            Tensor &t = TTOS; POP();
            MTOS.forward(t);
            mmu.free(t);                        /// * release dataset
        }
        else if (IS_M(top) && IS_OBJ(rs[-1])) { /// data set on rs[-1]
            Tensor &t = (Tensor&)mmu.du2obj(rs[-1]);
            if (t.is_dataset()) MTOS.forward(t);
            else ERROR("no dataset on RS?\n");
        }
        else ERROR("no dataset or tensor?\n")),
    CODE("backprop",
         if (TOS1T && IS_M(ss[-1])) {
             Tensor &t = TTOS; POP();
             MTOS.backprop(t);
             mmu.free(t);
         }
         else if (IS_M(top)) MTOS.backprop();
         else ERROR("model?\n")),
    CODE("predict",   {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        if (M1V) { DU t = POP(); MTOS.npush(t); }),
    CODE("n@",        if (M1V) { I16 i = POPi; PUSH(mmu.view(MTOS[i])); }),
    CODE("network",   if (IS_M(top)) fout << top),
    CODE("dataset",
        char *dsn = next_idiom();           ///< retrieve dataset name
        I16   bsz = POPi;                   ///< batch size
        PUSH(mmu.dataset(bsz));             /// * create a dataset as TOS
        fout << opx(OP_DATA, 0, top) << dsn;
        state = VM_WAIT),
    CODE("fetch",
        if (IS_OBJ(rs[-1])) {
            Dataset &d = (Dataset&)mmu.du2obj(rs[-1]);
            if (!d.is_dataset()) {
                ERROR("not a dataset on RS?\n"); return;
            }
            if (d.done) return;
            
            fout << opx(OP_LOAD, 0, rs[-1]);       /// * issue an reload
            state = VM_WAIT;                       /// * return to CPU
        }
        else ERROR("dataset?"))
    ///@}
    };
    const Code over[] = {                          ///< extended (overload) words
    CODE("donext",                                 /// * overwrite "donext" in eforth.cu
        if (IS_M(top) && IS_OBJ(rs[-1])) {
            Model   &m = (Model&)mmu.du2obj(top);
            Dataset &d = (Dataset&)mmu.du2obj(rs[-1]);
            if (!d.is_dataset()) {
                ERROR("not a dataset on RS?\n"); return;
            }
            if (d.done) {
                rs.pop();                          /// * pop off dataset
                mmu.free(d);                       /// * free the dataset
                IP += sizeof(IU);                  /// * skip over to next word
            }
            else {
                fout << opx(OP_LOAD, 0, rs[-1]);   /// * issue an reload
                state = VM_WAIT;                   /// * return to CPU
                IP    = mmu.ri(IP);                /// * loop branch target address
            }
        }
        else if ((rs[-1] -= 1) >= -DU_EPS) {
            IP = mmu.ri(IP);                      /// * handle numeric for loop
        }
        else { rs.pop(); IP += sizeof(IU); }),
    CODE("flatten",   nnop(L_FLATTEN)),
    CODE("boot",      mmu.clear(FIND("fetch") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code));   /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code));   /// * overload existed words
    
    VLOG1("NetVM::init ok\n");
};
#endif  // T4_ENABLE_OBJ
//===========================================================================
