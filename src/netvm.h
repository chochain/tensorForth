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

private:
    __GPU__ void _conv2d();       ///< convolution layer
    ///
    /// Batch ops
    ///
    __GPU__ void nn_for();
    __GPU__ void nn_next();
    __GPU__ void forward();
    __GPU__ void backprop();
    ///
    /// Gradiant ops
    ///
    __GPU__ void sgd();
    __GPU__ void adam();
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
