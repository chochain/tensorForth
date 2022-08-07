/** -*- c++ -*-
 * @File
 * @brief - Neural Network Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "netvm.h"

#if T4_ENABLE_OBJ
__GPU__ DU
NetVM::NPOP() {
    DU n = ntop;
    ntop = ((DU*)net.data)[--nidx];
    nten = &mmu.du2ten(ntop);
    return n;
}
__GPU__ DU
NetVM::NPUSH(DU v) {
    nten = &mmu.du2ten(v);
    return ((DU*)net.data)[nidx++] = (ntop = v);
}
__GPU__ DU
NetVM::NPUSH(Tensor &t) {
    nten = &t;
    return ((DU*)net.data)[nidx++] = (ntop = mmu.ten2du(t));
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
__GPU__ void
NetVM::dconv2d(Tensor &A, Tensor &B) {
}
__GPU__ void
NetVM::drelu(Tensor &A, Tensor &B) {
}
__GPU__ void
dmaxpool(Tensor &A, Tensor &B) {
}
__GPU__ void
dreshape(Tensor &A, Tensor &B) {
}
__GPU__ void
dlinear(Tensor &A, Tensor &B) {
}
///
/// Convolution ops
///
__GPU__ void
NetVM::conv2d() {
    const U16 opt[] = { 3, 3, 1, 1, 1 };
    conv2d((U16*)opt);
}
__GPU__ void
NetVM::conv2d(U16 *opt) {
    U16 c     = POPi;
    DU  bias  = POP();
    Tensor &t = *nten;                         ///> DAG tensor pointer
    if (!t.grad_fn) initgrad(t, bias, c, opt); ///> create tensors if not yet
    ///
    /// apply convolution filter
    ///
    NPUSH(t);
}
///
/// Activation ops
///
__GPU__ void
NetVM::relu() {
}
__GPU__ void
NetVM::tanh() {
}
__GPU__ void
NetVM::sigmoid() {
}
__GPU__ void
NetVM::softmax() {
}
///
/// Pooling ops
///
__GPU__ void
NetVM::meanpool(U16 n) {
}
__GPU__ void
NetVM::avgpool(U16 n) {
}
__GPU__ void
NetVM::maxpool(U16 n) {
}
__GPU__ void
NetVM::minpool(U16 n) {
}
///
/// Pooling ops
///
__GPU__ void
NetVM::linear(U16 n) {
}
///
/// Pooling ops
///
__GPU__ void
NetVM::dropout(U16 p) {
}
///
/// Back Propegation ops
///
__GPU__ void
NetVM::initgrad(Tensor &A, DU bias, U16 c, U16 *opt) {
    /// create autograd tensors
    ///
    A.grad_fn = &dconv2d;                        ///> derivative function

    U16 m = opt[0], n = opt[1];                  ///> filter sizing
    U16 p = opt[2] ? opt[2] : floor((m-1)/2);    ///> padding
    U16 s = opt[3], d = opt[4];                  ///> stride, dilation
    
    Tensor *w  = A.grad[0] = &mmu.tensor(1, m, n, c);                   ///> w
    mmu.random(*w, NORMAL);
    Tensor *b  = A.grad[1] = &mmu.tensor(1, 1, 1, c).map(FILL, bias); ///> b
    Tensor *dw = A.grad[2] = &mmu.tensor(1, m, n, c).map(FILL, DU0);  ///> dw
    Tensor *db = A.grad[3] = &mmu.tensor(1, 1, 1, c).map(FILL, DU0);  ///> db
}
__GPU__ void
NetVM::autograd(bool on) {
    f_autograd = on;
}
__GPU__ void
NetVM::for_batch() {
    Tensor &A = mmu.tensor(1, 28, 28, 1);
    NPUSH(A);
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
__GPU__ void
NetVM::network() {
    DU t = mmu.ten2du(net);
    printf("net.size = %d\n", net.size);
    DU *d = (DU*)net.data;
    for (int i = 0; i < nidx; i++) {
        DU     v   = *d++;
        Tensor &ti = mmu.du2ten(v);
        printf("%08x[%03d] => size=%d\n", *(U32*)&v, i, ti.size);
    }
}
///===================================================================
/// class methods
///
/// Neural Network specific dictionary constructor
///
__GPU__ void
NetVM::init() {
    const Code prim[] = {       /// singleton, build once only
    ///@defgroup Convolution ops
    ///@{
    CODE("conv2d",              ///> (Ta b c [A] -- Ta')
         if (IS_OBJ(top)) conv2d();
         else {
             Tensor &v = mmu.du2ten(top); POP();
             U16 opt[5];
             DU  *d = (DU*)v.data;
             for (int k=0; k<5; k++) opt[k] = (U16)*d++;
             conv2d(opt);
         }),
    ///@}
    ///@defgroup Activation ops
    ///@{
    CODE("relu",      {}),
    CODE("tanh",      {}),
    CODE("sigmoid",   {}),
    CODE("softmax",   {}),
    ///@}
    ///@defgroup Pooling ops
    ///@{
    CODE("meanpool",  {}),
    CODE("avgpool",   {}),
    CODE("maxpool",   {}),
    CODE("minpool",   {}),
    ///@}
    ///@defgroup Loss functions
    ///@{
    CODE("linear",    {}),
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
    CODE("autograd",  {}),
    ///@}
    ///@defgroup Debugging ops
    ///@{
//    CODE("network",  fout << opx(OP_NET, nidx, mmu.ten2du(net))),
    CODE("network",   network()),
    CODE(">n",        NPUSH(top); POP()),
    CODE("n>",        DU t = NPOP(); PUSH(t)),
    ///@}
    };
    const Code over[] = {          /// extended (overload) words
    CODE("boot", mmu.clear(FIND("autograd") + 1))
    };
    TensorVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(over,  sizeof(over)/sizeof(Code)); /// * overload existed words
    mmu.status();
};
#endif  // T4_ENABLE_OBJ
//=======================================================================================
