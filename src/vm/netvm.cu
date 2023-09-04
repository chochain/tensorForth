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
    /// handle tensor ops (destructive)
    ///
    if (TOS1T) {
        Tensor &t = TTOS;
        switch (op) {
        case L_FLATTEN: t.reshape(t.numel); return;
        case L_RELU:    t.map(O_RELU);      return;
        case L_TANH:    t.map(O_TANH);      return;
        case L_SIGMOID: t.map(O_SIGM);      return;
        case L_SOFTMAX:
            t.map(O_MUL, DU1 / (t.sum() + DU_EPS)); return;
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
    if (IS_M(top)) {
        Model &m = MTOS;
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
        switch (op) {
        case L_LINEAR:  m.add(op, INT(a));             return; /* bias = 0.0 */
        case L_LEAKYRL:
        case L_ELU:     
        case L_DROPOUT: m.add(op, 0, a);               return;
        case L_AVGPOOL:
        case L_MAXPOOL: 
        case L_MINPOOL: m.add(op, INT(a));             return;
        case L_BATCHNM: m.add(op, 0, a);               return;
        case L_USAMPLE: m.add(op, INT(a), UP_NEAREST); return;
        }
        PUSH(a);                                   /// * restore top
        /// continue to error handling cases
    }
    switch (op) {
    case L_LINEAR:
        if (M2V) {                                 /// * param checking
            U16 n    = POPi;                       ///> number of output channels
            DU  bias = POP();                      ///> linear bias
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
        if (!IS_OBJ(top)) {
            switch (op) {
            case L_RELU:    xop1(O_RELU, DU0); break;
            case L_TANH:    xop1(O_TANH);      break;
            case L_SIGMOID: xop1(O_SIGM);      break;
            }
        }
        else ERROR("NetVM::nnop layer %d not supported(2)\n", op);
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
    
    if (ss.idx > 1 && IS_OBJ(ss[-2])) { /* OK */ }
    else if (ss.idx > 2 && IS_OBJ(ss[-3])) mode |= POPi;       ///< TODO: RAW format
    else { ERROR("model/tensor adr len [mode]?\n"); return; }
    
    IU   len = POPi;                        ///< string length (not used for now)
    IU   adr = POPi;                        ///< address to pmem
    char *fn = (char*)mmu.pmem(adr);        ///< pointer to string on PAD
    _opx op  = IS_M(top)
        ? opx(save ? OP_NSAVE : OP_NLOAD, mode, top)
        : opx(OP_TSAVE, mode, top);
    fout << op << fn;                       /// * issue pickle command
    state = VM_WAIT;                                         /// * return to CPU
}

__GPU__ void
NetVM::_fetch(DU d, bool rewind) {
    if (!((Dataset&)mmu.du2obj(d)).is_dataset()) {
        ERROR("TOS=%08x not dataset?\n", DU2X(d));
        return;
    }
    fout << opx(OP_FETCH, rewind ? FAM_REW : 0, d);        /// * issue a fetch or rewind
    state = VM_WAIT;                                       /// * return to CPU
}
///
/// fetch parameters onto TOS
/// n=0:W, 1:B, 2:dW, 3:dB
///
__GPU__ void
NetVM::_parm(int n) {
    if (!M1V) { ERROR("N n required?"); return; }
    
    S16 i = POPi; Tensor *p = MTOS[i].grad[n]; 
    if (p) PUSH(mmu.copy(*p)); else PUSH(DU0); 
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
        Tensor &y = TTOS; POP();        /// * pop off target tensor
        Tensor &x = TTOS;
        PUSH(x.loss(op, y));
        mmu.mark_free(y);               /// * delayed free
    }
    else if (TOS1T && IS_M(ss[-1])) {
        Tensor &t = TTOS; POP();
        DU     n  = MTOS.loss(op, t);
        mmu.free(t);                    /// * pop off t
        PUSH(n);                        /// * loss on TOS
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
             MTOS.sgd(lr, DU0);               ///< default momentum = 0.0
         }
         else ERROR("rate mtum nn.sgd?\n")),
    CODE("nn.adam",
         if (M2V) {                           ///> (N lr b1 -- N')
             DU b1 = POP();                   ///< beta1 i.g. 0.9
             DU b2 = DU1 - POW(DU1 - b1, 3);  ///< default beta2 i.g. 0.999
             DU lr = POP();                   ///< learning rate i.g. 0.001
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
    CODE("trainable",  if (M1V) { bool on = POPi; MTOS.train = on; }),
    CODE("batchsize",
         if (IS_M(top)) PUSH(MTOS.batch_size());
         else ERROR("TOS is not a model?\n")),
    CODE("dataset",                             /// * create a dataset
        char *dsn = next_idiom();               ///< retrieve dataset name
        S16   bsz = POPi;                       ///< batch size
        PUSH(mmu.dataset(bsz));                 /// * create a dataset as TOS
        fout << opx(OP_DATA, 0, top) << dsn;    /// * issue a dataset init command
        state = VM_WAIT),
    CODE("fetch",   _fetch(top, false)),        /// * fetch a dataset batch
    CODE("rewind",  _fetch(top, true)),         /// * rewind a dataset (batch_id=0)
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
    CODE("broadcast",
        if (IS_M(ss[-1]) && TOS1T) {          /// * TOS is a onehot vector
            Tensor &t = TTOS; POP();
            MTOS.broadcast(t);
            mmu.free(t);
        }
        else ERROR("TOS not a tensor nor NOS a model?\n")),
    ///@}
    ///@defgroup Debugging ops
    ///@{
    CODE(">n",      if (M1V) { DU  t = POP(); MTOS.npush(t); }),
    CODE("n@",      if (M1V) { S16 i = POPi; PUSH(mmu.copy(MTOS[i])); }),
    CODE("nn.w",    _parm(0)),                     ///< tensor.weight
    CODE("nn.b",    _parm(1)),                     ///< tensor.bias
    CODE("nn.dw",   _parm(2)),                     ///< tensor.weight.grad
    CODE("nn.db",   _parm(3)),                     ///< tensor.bias.grad
    CODE("network", if (IS_M(top)) fout << top),
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
                m.epoch++;                         /// * bump epoch counter
                rs.pop();                          /// * pop off dataset
                IP += sizeof(IU);                  /// * skip over to next word
            }
            else {
                _fetch(rs[-1], false);             /// * issue a dataset fetch
                IP = mmu.ri(IP);                   /// * loop branch target address
            }
        }
        else if ((rs[-1] -= 1) >= -DU_EPS) {
            IP = mmu.ri(IP);                      /// * handle numeric for loop
        }
        else { rs.pop(); IP += sizeof(IU); }),
    CODE("flatten",   nnop(L_FLATTEN)),
    CODE("save",      _pickle(true)),             /// * save trainned model
    CODE("load",      _pickle(false)),            /// * load trainned model
    CODE("boot",      mmu.clear(FIND("network") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code));   /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code));   /// * overload existed words
    
    VLOG1("NetVM::init ok\n");
};
#endif  // T4_ENABLE_OBJ
//===========================================================================
