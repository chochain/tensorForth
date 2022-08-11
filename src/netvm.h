/**
 * @file
 * @brief tensorForth - NetVM, extended TensorVM classes, to handle Neural Network ops
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_NETVM_H
#define TEN4_SRC_NETVM_H
#include "model.h"
#include "tenvm.h"                /// extending TensorVM

#define NTOP  (mmu.du2mdl(top))   /// Network Model on TOS
class NetVM : public TensorVM {
public:
#if   !T4_ENABLE_OBJ
    __GPU__ NetVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        TensorVM(khz, istr, ostr, mmu0) {}
    __GPU__ void init() { TensorVM::init(); }

#else // T4_ENABLE_OBJ
    ///
    /// @name static Loss functions
    /// @{
    static __GPU__ void loss_nll(Tensor &A, Tensor &B, Tensor &C);  ///< negative likelihood
    static __GPU__ void loss_mse(Tensor &A, Tensor &B, Tensor &C);  ///< mean square error
    static __GPU__ void loss_ce(Tensor &A, Tensor &B, Tensor &C);   ///< cross entropy
    static __GPU__ void predict(Tensor &A, Tensor &B, Tensor &C);   ///< predict result
    /// @}
    /// @name Class Object Constructor
    /// @{
    __GPU__ NetVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        TensorVM(khz, istr, ostr, mmu0) {
        VLOG1("\\  ::NetVM(...) sizeof(Model)=%d\n", (int)sizeof(Model));
    }
    __GPU__ void init() final;    ///< override TensorVM, TODO: does not work without 'final'!

protected:
    ///
    /// @name Convolution and Linear ops
    /// @{
    __GPU__ void conv2d();        ///< convolution with bias and c output channel
    __GPU__ void linear();        ///< linearize with n output
    __GPU__ void flatten();       ///< flatten (fully connected)
    /// @}
    /// @name Activation ops
    /// @{
    __GPU__ void relu();          ///< Rectified Linear Unit
    __GPU__ void tanh();          ///< Tanh Unit
    __GPU__ void sigmoid();       ///< 1/(1+exp(-z))
    __GPU__ void softmax();       ///< probability vector exp(x)/sum(exp(x))
    /// @}
    /// @name Pooling and Dropout ops
    /// @{
    __GPU__ void maxpool();       ///< maximum pooling with nxn filter
    __GPU__ void avgpool();       ///< average pooling with nxn filter
    __GPU__ void minpool();       ///< minimum pooling with nxn filter
    __GPU__ void dropout();       ///< zero out p% of channel data (add noise between data points)
    /// @}
    ///
    /// Back propergation ops
    ///
    __GPU__ void autograd(bool on=false);
    __GPU__ void for_batch();
    __GPU__ void backprop();
    __GPU__ void sgd();
    __GPU__ void adam();
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
