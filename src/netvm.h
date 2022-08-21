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

#define NN0      ((Model&)mmu.du2obj(top))                         /** Network Model on TOS   */
#define NN1      ((Model&)mmu.du2obj(ss[-1]))                      /** Network Model on NOS   */
#define IS_M(v)  (IS_OBJ(v) && mmu.du2obj(v).is_model())           /** check param is a model */
#define MTOS     (IS_M(top))                                       /** TOS is a model         */
#define MNOS     (!IS_OBJ(top) && IS_M(ss[-1]))                    /** NOS model w 1-param    */
#define MN2D     (!IS_OBJ(top) && !IS_OBJ(ss[-1]) && IS_M(ss[-2])) /** ss[-2] model w 2-param */

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
    __GPU__ void nnop(t4_layer op);

private:
    __GPU__ void _conv();         ///< convolution layer
    ///
    /// Batch ops
    ///
    __GPU__ void nn_for();
    __GPU__ void nn_next();
    ///
    /// Gradiant ops
    ///
    __GPU__ void sgd();
    __GPU__ void adam();
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
