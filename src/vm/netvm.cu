/** -*- c++ -*-
 * @file
 * @brief NetVM class - extend TensorVM class, Neural Network Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "dataset.h"           // in ../mmu
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
    case L_CONV:  break;                           ///> conv handled at VM level
    case L_LINEAR:
        if (M2V) {                                 ///> param checking
            U16   n    = POPi;                     ///> number of output channels
            DU    bias = POP();                    ///> linear bias
            MTOS.add(L_LINEAR, n, bias);           ///> (N b c -- N')
        }
        else if (M1V) {
            U16   n    = POPi;                     ///> number of output channels
            DU    bias = DU0;                      ///> linear bias=0.0
            MTOS.add(L_LINEAR, n, bias);           ///> (N c -- N')
        }
        else ERROR("linear: [bias] n required!");
        break;
    case L_FLATTEN:
    case L_RELU:
    case L_TANH:
    case L_SELU:
    case L_SIGMOID: if (IS_M(top)) MTOS.add(op); break;
    case L_LEAKYRL:
    case L_ELU:     if (M1V) { DU  a = POP(); MTOS.add(op, 0, a); } break;
    case L_SOFTMAX:
    case L_LOGSMAX: if (IS_M(top)) MTOS.add(op); break;
    case L_MAXPOOL: 
    case L_AVGPOOL:
    case L_MINPOOL: if (M1V) { U16 n = POPi;  MTOS.add(op, n); }    break;
    case L_DROPOUT: if (M1V) { DU  p = POP(); MTOS.add(op, 0, p); } break;
    case L_USAMPLE: {
        U16 n = POPi;
        U16 m = (M1V) ? POPi : UP_NEAREST;
        MTOS.add(op, n, m);
    } break;
    case L_BATCHNM:
        if (IS_M(top)) MTOS.add(op, 0, 0.1);   /// * default momentum=0.1
        else if (M1V) {
            DU m = POP(); MTOS.add(op, 0, m);
        }
        break;
    default: ERROR("NetVM::nnop(%d) not supported\n", op);
    }
}

__GPU__ void
NetVM::predict(Tensor &I, Tensor &P) {
}
///===================================================================
/// private methods
///
///
/// dataset ops
///
__GPU__ void
NetVM::_pickle(bool save) {
    IU   mode= save ? FAM_WO : FAM_RW;      ///< file access mode
    
    if (ss.idx > 1 && IS_M(ss[-2])) { /* OK */ }
    else if (ss.idx > 2 && IS_M(ss[-3])) mode |= POPi;       ///< TODO: RAW format
    else { ERROR("model adr len [mode]?\n"); return; }
    
    IU   len = POPi;                        ///< string length (not used for now)
    IU   adr = POPi;                        ///< address to pmem
    char *fn = (char*)mmu.pmem(adr);        ///< pointer to string on PAD
    fout << opx(save ? OP_NSAVE : OP_NLOAD, mode, top) << fn;/// * issue pickle command
    state = VM_WAIT;                                         /// * return to CPU
}

__GPU__ void
NetVM::_fetch(DU d, bool more) {
    if (!((Dataset&)mmu.du2obj(d)).is_dataset()) {
        ERROR("TOS=%08x not dataset?\n", DU2X(d));
        return;
    }
    fout << opx(OP_FETCH, (U16)more, d);                   /// * issue a fetch or rewind
    state = VM_WAIT;                                       /// * return to CPU
}
/// Convolution ops
/// @default: kxk filter, padding=1, stride=1, dilation=1
/// @parameters
///    k: kernel size
///
__GPU__ void
NetVM::_conv(U16 k) {
    U16 opt[] = { k, k, 1, 1, 1 };      ///> default config vector
    if (TOS1T) {                        ///> if optional vector given
        Tensor &v = TTOS;
        if (v.rank == 1) {
            for (int i=0; i<5; i++) opt[i] = (U16)v.data[i];
            POP(); mmu.free(v);
        }
        else { ERROR("vec?"); return; }
    }
    if (!M2V) { ERROR("convolution: bias c required!"); return; }
    U16 c    = POPi;                    ///> number of output channels
    DU  bias = POP();                   ///> convolution bias
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
        mmu.free(t);                    /// * pop off t
        PUSH(n);                        /// * loss on TOS
        VLOG1("NetVM#loss => %.3f\n", n);
    }
    else if (IS_M(top)) PUSH(MTOS.loss(op));
    else ERROR("model?\n");
}
///===================================================================
///
/// Neural Network Vocabulary
///
__GPU__ void
NetVM::init() {
    const Code prim[] = {                   ///> singleton, build once only
    ///@defgroup Model creation and persistence
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
    CODE("nn.save",   _pickle(true)),         /// * save trainned model
    CODE("nn.load",   _pickle(false)),        /// * load trainned model
    ///@}
    ///@defgroup Convolution and Linear ops
    ///@{
    CODE("conv1x1",   _conv(1)),              ///> (N b c -- N')
    CODE("conv2d",    _conv(3)),              ///> (N b c [A] -- N')
    CODE("linear",    nnop(L_LINEAR)),        ///> (N b c -- N')
    ///@}
    ///@defgroup BatchNorm and Activation ops
    ///@{
    CODE("relu",      nnop(L_RELU)),          ///> (N -- N')
    CODE("tanh",      nnop(L_TANH)),          ///> (N -- N')
    CODE("sigmoid",   nnop(L_SIGMOID)),       ///> (N -- N')
    CODE("selu",      nnop(L_SELU)),          ///> (N -- N')
    CODE("leakyrelu", nnop(L_LEAKYRL)),       ///> (N a -- N')
    CODE("elu",       nnop(L_ELU)),           ///> (N a -- N')
    CODE("softmax",   nnop(L_SOFTMAX)),       ///> (N -- N')
    CODE("logsoftmax",nnop(L_LOGSMAX)),       ///> (N -- N')
    CODE("batchnorm", nnop(L_BATCHNM)),       ///> (N -- N')
    ///@}
    ///@defgroup Pooling, Dropout, and Upsample ops
    ///@{
    CODE("maxpool",   nnop(L_MAXPOOL)),       ///> (N n -- N')
    CODE("avgpool",   nnop(L_AVGPOOL)),       ///> (N n -- N')
    CODE("minpool",   nnop(L_MINPOOL)),       ///> (N n -- N')
    CODE("dropout",   nnop(L_DROPOUT)),       ///> (N p -- N')
    CODE("upsample",  nnop(L_USAMPLE)),       ///> (N [m] n -- N')
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("loss.mse",  _loss(LOSS_MSE)),       ///> (N T -- N T n) mean square error
    CODE("loss.bce",  _loss(LOSS_BCE)),       ///> (N T -- N T n) binary cross-entropy
    CODE("loss.ce",   _loss(LOSS_CE)),        ///> (N T -- N T n) cross-entropy
    CODE("loss.nll",  _loss(LOSS_NLL)),       ///> (N T -- N T n) negative log-likelihood
    CODE("nn.loss",                           ///> (N T -- N T n) auto select loss function
         if (IS_M(top) || (TOS1T && IS_M(ss[-1]))) {
             Model &m = IS_M(top) ? MTOS : (Model&)mmu.du2obj(ss[-1]);
             switch (m[-2].grad_fn) {
             case L_TANH:
             case L_SIGMOID: _loss(LOSS_BCE); break;
             case L_SOFTMAX: _loss(LOSS_CE);  break;
             case L_LOGSMAX: _loss(LOSS_NLL); break;
             default:        _loss(LOSS_MSE);
             }
         }
         else ERROR("TOS is not a tensor or NOS is not a model!\n")),
    ///@}
    ///@defgroup Gradiant ops
    ///@{
    CODE("nn.zero",
         if (IS_M(top)) MTOS.grad_zero();
         else ERROR("TOS is not a model!\n")),
    CODE("nn.sgd",                            
         if (M2V) {                           ///> (N p m -- N')
             DU m  = POP();                   ///< momentum
             DU lr = POP();                   ///< learn rate
             MTOS.sgd(lr, m);
         }
         else if (M1V) {                      ///> (N p -- N')
             DU lr = POP();                   ///< learn rate
             MTOS.sgd(lr, 0.0);               ///< default momentum = 0.0
         }
         else ERROR("rate mtum nn.sgd?\n")),
    CODE("nn.adam",
         if (M2V) {                           ///> (N lr b1 -- N')
             DU b1 = POP();                   ///< beta1
             DU b2 = DU1 - POW(DU1 - b1, 3);  ///< default beta2
             DU lr = POP();                   ///< learn rate
             MTOS.adam(lr, b1, b2);
         }
         else ERROR("rate beta1 nn.adam?\n")),
    CODE("nn.onehot",                         /// * current onehot vector
        if (IS_M(top)) {
            Tensor &hot = MTOS.onehot();
            PUSH(hot); hot.ref_inc();
        }
        else ERROR("TOS is not a model!\n")),
    CODE("nn.hit", 
        if (IS_M(top)) PUSH(I2D(MTOS.hit()));
        else ERROR("TOS is not a model!\n")),
    ///@}
    ///@defgroup Batch Control ops
    ///@{
    CODE("autograd",  if (M1V) { bool on = POPi; MTOS.autograd = on; }),
    CODE("batchsize",
         if (IS_M(top)) PUSH(MTOS.batch_size());
         else ERROR("TOS is not a model?\n")),
    CODE("dataset",                             /// * create a dataset
        char *dsn = next_idiom();               ///< retrieve dataset name
        I16   bsz = POPi;                       ///< batch size
        PUSH(mmu.dataset(bsz));                 /// * create a dataset as TOS
        fout << opx(OP_DATA, 1, top) << dsn;    /// * issue a dataset init command
        state = VM_WAIT),
    CODE("fetch",   _fetch(top, true)),         /// * fetch a dataset batch
    CODE("rewind",  _fetch(top, false)),        /// * rewind a dataset (batch_id=0)
    CODE("forward",                             /// * forward process
        if (IS_M(ss[-1]) && TOS1D) {            /// * TOS is a dataset
            Tensor &t = TTOS; POP();            /// * NOS is the model
            MTOS.forward(t);                    /// * exec forward path
            mmu.free(t);                        /// * release reference
        }
        else if (IS_M(top) && IS_OBJ(rs[-1])) {       /// * in a for/next loop
            Tensor &t = (Tensor&)mmu.du2obj(rs[-1]);  /// * rs[-1] is a dataset
            if (t.is_dataset()) MTOS.forward(t);
            else ERROR("rs[-1] is not a dataset?\n");
        }
        else ERROR("no model or a dataset?\n")),
    CODE("backprop",
         if (IS_M(ss[-1]) && TOS1T) {          /// * TOS is a onehot vector
             Tensor &t = TTOS; POP();
             MTOS.backprop(t);
             mmu.free(t);
         }
         else if (IS_M(top)) MTOS.backprop();  /// * use default output
         else ERROR("TOS not a model?\n")),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",        if (M1V) { DU t = POP(); MTOS.npush(t); }),
    CODE("n@",        if (M1V) { I16 i = POPi; PUSH(mmu.view(MTOS[i])); }),
    CODE("network",   if (IS_M(top)) fout << top),
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
                m.grad_zero();                     /// * reset momentum tensors
                rs.pop();                          /// * pop off dataset
                IP += sizeof(IU);                  /// * skip over to next word
            }
            else {
                _fetch(rs[-1], true);              /// * issue a dataset fetch
                IP = mmu.ri(IP);                   /// * loop branch target address
            }
        }
        else if ((rs[-1] -= 1) >= -DU_EPS) {
            IP = mmu.ri(IP);                      /// * handle numeric for loop
        }
        else { rs.pop(); IP += sizeof(IU); }),
    CODE("flatten",   nnop(L_FLATTEN)),
    CODE("boot",      mmu.clear(FIND("network") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code));   /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code));   /// * overload existed words
    
    VLOG1("NetVM::init ok\n");
};
#endif  // T4_ENABLE_OBJ
//===========================================================================
