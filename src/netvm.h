/**
 * @file
 * @brief tensorForth - NetVM, extended TensorVM classes, to handle Neural Network ops
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_NETVM_H
#define TEN4_SRC_NETVM_H
#include "tenvm.h"                         /// extending TensorVM

class NetVM : public TensorVM {
public:
#if   !T4_ENABLE_OBJ
    __GPU__ NetVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        TensorVM(khz, istr, ostr, mmu0) {}
    __GPU__ void init() { TensorVM::init(); }
    
#else // T4_ENABLE_OBJ
    /// @name static Loss functions
    /// @{
    static __GPU__ void loss_nll(Tensor &A, Tensor &B, Tensor &C);  ///< negative likelihood
    static __GPU__ void loss_mse(Tensor &A, Tensor &B, Tensor &C);  ///< mean square error
    static __GPU__ void loss_ce(Tensor &A, Tensor &B, Tensor &C);   ///< cross entropy
    static __GPU__ void predict(Tensor &A, Tensor &B, Tensor &C);   ///< predict result
    /// @}
    /// @name Derivertive ops
    /// @{
    static __GPU__ void dconv2d(Tensor &A, Tensor &B);
    static __GPU__ void drelu(Tensor &A, Tensor &B);
    static __GPU__ void dmaxpool(Tensor &A, Tensor &B);
    static __GPU__ void dreshape(Tensor &A, Tensor &B);
    static __GPU__ void dlinear(Tensor &A, Tensor &B);
    /// @}
    /// @name Class Object Constructor
    /// @{
    __GPU__ NetVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        TensorVM(khz, istr, ostr, mmu0), net(mmu0->tensor(T4_NET_SZ)) {
        nidx = 0;
        ntop = mmu.ten2du(net);
        nten = &net;
        NPUSH(ntop);              ///< net tensor at [0] as stopper
        VLOG1("\\  ::NetVM(...) net=0x%08x(%.8f)\n", *(U32*)&ntop, ntop);
    }
    __GPU__ void init() final;    ///< override TensorVM, TODO: does not work without 'final'!
    
protected:
    Tensor  &net;                 ///< Neural Network DAG, TODO: tree structure
    U16     nidx;                 ///< network tensor index
    DU      ntop;                 ///< cached DAG tensor 
    Tensor  *nten;                ///< cached tensor pointer
    bool    f_autograd = false;   ///< autograd control flag
    /// @}
    /// @name Backprop DAG management short hands
    /// @{
    __GPU__ DU NPOP();
    __GPU__ DU NPUSH(DU v);
    __GPU__ DU NPUSH(Tensor &t);
    /// @}
    /// @name Convolution ops
    /// @{
    __GPU__ void conv2d();        ///< 2d convolution with bias=top, c channel output=ss[-1], 3x3 filter, padding=same, stride=1, dilation=1
    __GPU__ void conv2d(U16 *opt);///< 2d convolution with bias=top, c channel output=ss[-1], opt vector i.e. [5 5 3 2 1] for 5x5 filter, padding=3, stride=2, dilation=1
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ void relu();          ///< Rectified Linear Unit
    __GPU__ void tanh();          ///< Tanh Unit
    __GPU__ void sigmoid();       ///< 1/(1+exp(-z))
    __GPU__ void softmax();       ///< probability vector exp(x)/sum(exp(x))
    /// @}
    /// @name Pooling ops
    /// @{
    __GPU__ void meanpool(U16 n); ///< mean pooling with nxn filter
    __GPU__ void avgpool(U16 n);  ///< average pooling with nxn filter
    __GPU__ void maxpool(U16 n);  ///< maximum pooling with nxn filter
    __GPU__ void minpool(U16 n);  ///< minimum pooling with nxn filter
    /// @}
    /// @name Linear ops
    /// @{
    __GPU__ void linear(U16 n);   ///< linearize with n output
    /// @}
    /// @name Dropout ops
    /// @{
    __GPU__ void dropout(U16 p);  ///< zero out p% of channel data (add noise between data points)
    /// @}
    /// @name Back Propergation ops
    /// @{
    __GPU__ void initgrad(Tensor &A, DU bias, U16 c, U16 *opt);
    __GPU__ void autograd(bool on=false);
    __GPU__ void for_batch();
    __GPU__ void backprop();
    __GPU__ void sgd();
    __GPU__ void adam();
    /// @}
    __GPU__ void network();
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
