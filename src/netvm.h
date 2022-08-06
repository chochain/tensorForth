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
    __GPU__ void init_n() { TensorVM::init_t(); }
    
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
        TensorVM(khz, istr, ostr, mmu0) {
        VLOG1("\\  ::NetVM(...)\n");
    }
    __GPU__ void init() final;              ///< override TensorVM, TODO: does not work without 'final'!
    __GPU__ int  pre(char *idiom) {         ///< override vm.h
        ttop = IS_TEN(top) ? &mmu.du2ten(top) : NULL;
        return 0;
    }
    
protected:
    Tensor  *ttop = 0;                      ///< cached tensor on TOS
    
    ///
    /// @name Convolution ops
    /// @{
    __GPU__ void conv2d(DU bias, U16 c);           ///< 2d convolution with c channel output, 3x3 filter, padding=same, stride=1, dilation=1
    __GPU__ void conv2d(DU bias, U16 c, U16 *opt); ///< 2d convolution with c channel output, config vector V i.e. [5 5 2 0 1] for 3x3 filter, stride=1, padding=0, dilation=1
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
    __GPU__ void batch_for();
    __GPU__ void backprop();
    __GPU__ void sgd();
    __GPU__ void adam();
    /// @}
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
