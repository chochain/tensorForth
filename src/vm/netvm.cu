/** -*- c++ -*-
 * @file
 * @brief NetVM class - extend TensorVM class, Neural Network Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "netvm.h"

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)

__GPU__ void
NetVM::predict(Tensor &I, Tensor &P) {}

///===================================================================
/// private methods
///
__GPU__ void
NetVM::_nnop(t4_layer op) {     /// vtable dispatcher
    OPN(LAYER_OP);
    ///
    /// handle tensor ops (destructive)
    ///
    VLOG2("netvm#nnop %s", opn[op]);
    if (TOS1T) {
        Tensor &t = TTOS;
        VLOG2(" T%d", t.rank);
        switch (op) {
        case L_FLATTEN: t.reshape(t.numel); return;
        case L_RELU:    t.map(RELU);        return;
        case L_TANH:    t.map(TANH);        return;
        case L_SIGMOID: t.map(SIGM);        return;
        case L_SOFTMAX:
            t.map(MUL, RCP(t.sum() + DU_EPS)); return;
        case L_LOGSMAX:
            DU sum = t.sum();
            if (sum > DU_EPS) t -= LOG(sum);
            else ERROR("logsoftmax tensor sum < 0!");
            return;
        }
        // * continue to zero param
    }
    ///
    /// zero parameter layers
    ///
    if (IS_M(tos)) {
        Model &m = MTOS;
        VLOG2(" N%d {\n", m.numel);
        switch (op) {
        case L_FLATTEN:
        case L_RELU:
        case L_TANH:
        case L_SIGMOID:
        case L_SELU:    m.add(op);           return;
        case L_LEAKYRL: m.add(op, 0, 0.01);  return;
        case L_ELU:     m.add(op, 0, DU1);   return;
        case L_SOFTMAX:
        case L_LOGSMAX: m.add(op);           return;
        case L_BATCHNM: m.add(op, 0, 0.1);   return; /// * default momentum=0.1
        }
        // * continue to one param
    }
    ///
    /// one parameter layers
    ///
    if (M1V) {
        DU    a  = POP();
        Model &m = MTOS;
        VLOG2(" N%d %g {\n", m.numel, a);
        switch (op) {
        case L_LINEAR:  m.add(op, INT(a), DU1);        return; /* bias = 1.0 */
        case L_LEAKYRL:
        case L_ELU:     
        case L_DROPOUT: m.add(op, 0, a);               return;
        case L_AVGPOOL:
        case L_MAXPOOL: 
        case L_MINPOOL: m.add(op, INT(a));             return;
        case L_BATCHNM: m.add(op, 0, a);               return;
        case L_USAMPLE: m.add(op, INT(a), UP_NEAREST); return;
        }
        PUSH(a);                                   /// * restore tos
        /// continue to error handling cases
    }
    switch (op) {
    case L_LINEAR:
        if (M2V) {                                 /// * param checking
            U16 n    = POPi;                       ///> number of output channels
            DU  bias = POP();                      ///> bias range [-bias, bias)
            MTOS.add(op, n, bias);                 /// * (N b c -- N')
        }
        else ERROR("( N [bias] n -- ) for linear required!");
        break;
    case L_FLATTEN:
    case L_SELU:
    case L_SOFTMAX:
    case L_LOGSMAX: ERROR("( N -- ) no param needed!"); break;
    case L_LEAKYRL:
    case L_ELU:
    case L_DROPOUT:
    case L_AVGPOOL:
    case L_MAXPOOL: 
    case L_MINPOOL:
    case L_BATCHNM: ERROR("( N n -- ) one param required!"); break;
    case L_USAMPLE:
        if (M2V) {
            U16 n = POPi;
            DU  m = POP();
            MTOS.add(op, n, m);
        }
        else ERROR("( N [mtum] n -- ) for upsample required?");
        break;
    default:
        if (!IS_OBJ(tos)) {
            switch (op) {
            case L_RELU:    xop1(RELU, DU0); break;
            case L_TANH:    xop1(TANH);      break;
            case L_SIGMOID: xop1(SIGM);      break;
            }
        }
        else ERROR("NetVM::nnop layer %d not supported(2)\n", op);
    }
    VLOG2("} netvm#nnop\n");
}
///
/// dataset ops
///
__GPU__ void
NetVM::_pickle(bool save) {
    U8   mode= save ? FAM_WO : FAM_RW;      ///< file access mode
    
    if (ss.idx > 1 && IS_OBJ(ss[-2])) { /* OK */ }
    else if (ss.idx > 2 && IS_OBJ(ss[-3])) mode |= POPi;       ///< TODO: RAW format
    else { ERROR("model/tensor adr len [mode]?\n"); return; }
    
    IU   len = POPi;                        ///< string length (not used for now)
    IU   adr = POPi;                        ///< address to pmem
    char *fn = (char*)mmu.pmem(adr);        ///< pointer to string on PAD
    sys.op(IS_M(tos) ? (save ? OP_NSAVE : OP_NLOAD) : OP_TSAVE, mode, tos);
    state = HOLD;                           /// * return to CPU
}

///
/// fetch parameters onto TOS
/// n=0:W, 1:B, 2:dW, 3:dB
///
__GPU__ void
NetVM::_get_parm(int n) {
    if (!M1V) { ERROR("N n required?"); return; }
    
    S16 i = POPi;
    Tensor *p = MTOS[i].grad[n];
    if (p) {
        DU v = mmu.obj2du(*p);
        PUSH(DUP(v));
    }
    else PUSH(DU0);
}
///
/// fetch parameters onto TOS
/// n=0:W, 1:B, 2:dW, 3:dB
///
__GPU__ void
NetVM::_set_parm(int n) {
    if (!MTV) { ERROR("N T n required?"); return; }

    S16    i  = POPi;
    Tensor &p = *MNOS[i].grad[n];
    Tensor &t = TTOS;
    if (t.numel == p.numel) {
        Tensor::copy(t, p);
        DU t = POP(); DROP(t);
    }
    else {
        PUSH(i);                        /// * restore n
        ERROR("Tensor and model parameter is not the same shape");
    }
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
            DU t = POP(); DROP(t);
        }
        else { ERROR("vec?"); return; }
    }
    if (!M2V) { ERROR("Model#add bias c for conv2d required!"); return; }
    U16 c    = POPi;                    ///> number of output channels
    DU  bias = POP();                   ///> convolution bias
    MTOS.add(L_CONV, c, bias, opt);
}
///
/// loss functions
///
__GPU__ void
NetVM::_loss(t4_loss op) {
    if (TOS2T) {                        /// * calculate loss of two tensors
        DU y = POP();                   /// * pop off target tensor
        DU n = TTOS.loss(op, (Tensor&)T4Base::du2obj(y));
        PUSH(n);
        DROP(y);                        /// * free target tensor
    }
    else if (TOS1T && IS_M(ss[-1])) {   /// * model loss
        DU y = POP();
        DU n = MTOS.loss(op, (Tensor&)mmu::du2obj(y));
        PUSH(n);                        /// * loss on TOS
        DROP(y);                        /// * pop off t
    }
    else if (IS_M(tos)) PUSH(MTOS.loss(op));
    else ERROR("model?\n");
}
///===================================================================
///
/// Neural Network Vocabulary
///
__GPU__ void
NetVM::init() {
    if (id!=0) return;                        /// * singleton
    TensorVM::init();
    ///
    ///@defgroup Model creation and persistence
    ///@{
    CODE("nn.model",                          ///> (n h w c -- N)
         if (ss.idx < 4 ||                    /// * param check
             IS_OBJ(tos) || IS_OBJ(ss[-1]) ||
             IS_OBJ(ss[-2]) || IS_OBJ(ss[-3])) {
             ERROR("n h w c?\n"); return;
         }
         U16 c=POPi; U16 w=POPi; U16 h=POPi; U16 n=POPi;
         Model  &m = mmu.model();             /// * create NN model
         Tensor &t = mmu.tensor(n,h,w,c);     /// * create input tensor
         m.npush(t);                          /// * serves as the 1st layer
         PUSH(m));
    ///@}
    ///@defgroup Convolution and Linear ops
    ///@{
    CODE("conv1x1",   _conv(1));              ///> (N b c -- N')
    CODE("conv2d",    _conv(3));              ///> (N b c [A] -- N')
    CODE("linear",    _nnop(L_LINEAR));       ///> (N b c -- N')
    ///@}
    ///@defgroup BatchNorm and Activation ops
    ///@{
    CODE("relu",      _nnop(L_RELU));         ///> (N -- N')
    CODE("tanh",      _nnop(L_TANH));         ///> (N -- N')
    CODE("sigmoid",   _nnop(L_SIGMOID));      ///> (N -- N')
    CODE("selu",      _nnop(L_SELU));         ///> (N -- N')
    CODE("leakyrelu", _nnop(L_LEAKYRL));      ///> (N a -- N')
    CODE("elu",       _nnop(L_ELU));          ///> (N a -- N')
    CODE("softmax",   _nnop(L_SOFTMAX));      ///> (N -- N')
    CODE("logsoftmax",_nnop(L_LOGSMAX));      ///> (N -- N')
    CODE("batchnorm", _nnop(L_BATCHNM));      ///> (N -- N')
    ///@}
    ///@defgroup Pooling, Dropout, and Upsample ops
    ///@{
    CODE("maxpool",   _nnop(L_MAXPOOL));      ///> (N n -- N')
    CODE("avgpool",   _nnop(L_AVGPOOL));      ///> (N n -- N')
    CODE("minpool",   _nnop(L_MINPOOL));      ///> (N n -- N')
    CODE("dropout",   _nnop(L_DROPOUT));      ///> (N p -- N')
    CODE("upsample",  _nnop(L_USAMPLE));      ///> (N [m] n -- N')
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("loss.mse",  _loss(LOSS_MSE));       ///> (N T -- N T n) mean square error
    CODE("loss.bce",  _loss(LOSS_BCE));       ///> (N T -- N T n) binary cross-entropy
    CODE("loss.ce",   _loss(LOSS_CE));        ///> (N T -- N T n) cross-entropy
    CODE("loss.nll",  _loss(LOSS_NLL));       ///> (N T -- N T n) negative log-likelihood
    CODE("nn.loss",                           ///> (N T -- N T n) auto select loss function
         if (IS_M(tos) || (TOS1T && IS_M(ss[-1]))) {
             Model &m = IS_M(tos) ? MTOS : (Model&)mmu.du2obj(ss[-1]);
             switch (m[-2].grad_fn) {
             case L_TANH:
             case L_SIGMOID: _loss(LOSS_BCE); break;
             case L_SOFTMAX: _loss(LOSS_CE);  break;
             case L_LOGSMAX: _loss(LOSS_NLL); break;
             default:        _loss(LOSS_MSE);
             }
         }
         else ERROR("TOS is not a tensor or NOS is not a model!\n"));
    ///@}
    ///@defgroup Gradiant ops
    ///@{
    CODE("nn.zero",
         if (IS_M(tos)) MTOS.grad_zero();
         else ERROR("TOS is not a model!\n"));
    CODE("nn.sgd",                            
         if (M2V) {                           ///> (N p m -- N')
             DU m  = POP();                   ///< momentum
             DU lr = POP();                   ///< learn rate
             MTOS.sgd(lr, m);
         }
         else if (M1V) {                      ///> (N p -- N')
             DU lr = POP();                   ///< learn rate
             MTOS.sgd(lr, DU0);               ///< default momentum = 0.0
         }
         else ERROR("rate mtum nn.sgd?\n"));
    CODE("nn.adam",
         if (M1V) {                           ///> (N lr -- N')
             DU lr = POP();                   /// * learing rate 
             MTOS.adam(lr, 0.9, 0.999);       /// * default b1=0.9, b2=0.999
         }
         else if (M2V) {                      ///> (N lr b1 -- N')
             DU b1 = POP();                   ///< beta1 i.g. 0.9
             DU lr = POP();                   ///< learning rate i.g. 0.001
             MTOS.adam(lr, b1, 0.999);
         }
         else ERROR("rate beta1 nn.adam?\n"));
    CODE("nn.onehot",                         /// * current onehot vector
         if (IS_M(tos)) {
             Tensor &hot = MTOS.onehot();
             DU v = mmu.obj2du(hot);
             PUSH(DUP(v));
         }
         else ERROR("TOS is not a model!\n"));
    CODE("nn.hit", 
         if (IS_M(tos)) PUSH(I2D(MTOS.hit()));
         else ERROR("TOS is not a model!\n"));
    ///@}
    ///@defgroup Batch Control ops
    ///@{
    CODE("trainable",
         if (M1V) { bool on = POPi; MTOS.train = on; }
         else ERROR("N [1|0] required\n"));
    CODE("batchsize",
         if (IS_M(tos)) PUSH(MTOS.batch_size());
         else ERROR("TOS is not a model?\n"));
    CODE("dataset",                             /// * create a dataset
         char *dsn = next_idiom();              ///< retrieve dataset name
         S16   bsz = POPi;                      ///< batch size
         PUSH(mmu.dataset(bsz));                /// * create a dataset as TOS
         sys.op(OP_DATA, 0, tos);
         sys.op_fn(dsn);                        /// * issue a dataset init command
         state = HOLD);
    CODE("fetch",   _fetch(tos, false));        /// * fetch a dataset batch
    CODE("rewind",  _fetch(tos, true));         /// * rewind a dataset (batch_id=0)
    CODE("forward",                             /// * forward process
         if (IS_M(ss[-1]) && TOS1D) {           /// * TOS is a dataset
             DU x = POP();                      /// * NOS is the model
             MTOS.forward((Tensor&)mmu.du2obj(x));     /// * exec forward path
             DROP(x);                           /// * release reference
         }
         else if (IS_M(tos) && IS_OBJ(rs[-1])) {       /// * in a for/next loop
             Tensor &t = (Tensor&)mmu.du2obj(rs[-1]);  /// * rs[-1] is a dataset
             if (t.is_dataset()) MTOS.forward(t);
             else ERROR("rs[-1] is not a dataset?\n");
         }
         else ERROR("no model or a dataset?\n"));
    CODE("backprop",
         if (IS_M(ss[-1]) && TOS1T) {                  /// * TOS is a onehot vector
             DU y = POP();                     
             MTOS.backprop((Tensor&)mmu.du2obj(y));    /// * backprop(target vector)
             DROP(y);
         }
         else if (IS_M(tos)) MTOS.backprop();          /// * use default output
         else ERROR("TOS not a model?\n"));
    CODE("broadcast",
         if (IS_M(ss[-1]) && TOS1T) {                  /// * TOS is a onehot vector
             DU y = POP();
             MTOS.broadcast((Tensor&)mmu.du2obj(y));
             DROP(y);
         }
         else ERROR("TOS not a tensor nor NOS a model?\n"));
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",      if (M1V) { DU t = POP(); MTOS.npush(t); });
    CODE("n@",      if (!M1V) return;
         S16    i  = POPi;
         Tensor &t = MTOS[i];
         DU     v  = mmu.obj2du(t);
         PUSH(DUP(v)));
    CODE("nn.w",    _get_parm(0));                 ///< tensor.weight
    CODE("nn.b",    _get_parm(1));                 ///< tensor.bias
    CODE("nn.dw",   _get_parm(2));                 ///< tensor.weight.grad
    CODE("nn.db",   _get_parm(3));                 ///< tensor.bias.grad
    CODE("nn.w=",   _set_parm(0));                 ///< populate tensor.weight
    CODE("nn.b=",   _set_parm(1));                 ///< populate tensor.bias
    CODE("network", if (IS_M(tos)) fout << tos);
    ///
    /// ===========================================================================
    ///
    /// * overwrite/extended word
    ///
    CODE("boot",      mmu.clear(FIND("network") + 1));
    CODE("donext",    _donext());                  /// * overwrite eforth.cu
    CODE("flatten",   _nnop(L_FLATTEN));
    CODE("save",      _pickle(true));              /// * save trainned model
    CODE("load",      _pickle(false));             /// * load trainned model
    
    VLOG1("NetVM::init ok\n");
};

#endif  // (T4_ENABLE_OBJ && T4_ENABLE_NN)
//===========================================================================
