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

#define MTOS     ((Model&)mmu.du2obj(top))                         /** Network Model on TOS   */
#define MNOS     ((Model&)mmu.du2obj(ss[-1]))                      /** Network Model on NOS   */
#define IS_M(v)  (IS_OBJ(v) && mmu.du2obj(v).is_model())           /** check param is a model */
#define M1V      (!IS_OBJ(top) && IS_M(ss[-1]))                    /** NOS model w 1-param    */
#define M2V      (!IS_OBJ(top) && !IS_OBJ(ss[-1]) && IS_M(ss[-2])) /** ss[-2] model w 2-param */

class NetVM : public TensorVM {
public:
#if   !T4_ENABLE_OBJ
    __GPU__ NetVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        TensorVM(khz, istr, ostr, mmu0) {}
    __GPU__ void init() { TensorVM::init(); }

#else // T4_ENABLE_OBJ
    ///
    /// @name Class Object Constructor
    /// @{
    __GPU__ NetVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        TensorVM(khz, istr, ostr, mmu0) {
        VLOG1("\\  ::NetVM(...) sizeof(Model)=%d\n", (int)sizeof(Model));
    }
    __GPU__ void init() final;    ///< override TensorVM, TODO: does not work without 'final'!
    __GPU__ void nnop(t4_layer op);
    __GPU__ void predict(Tensor &I, Tensor &P);   ///< predict result

private:
    ///
    /// @name Batch ops
    /// @{
    __GPU__ void nn_for();
    __GPU__ void nn_next();
    /// @}
    /// @name static Loss functions
    /// @{
    /// @}
    /// @name Convolution, loss and Gradiant ops
    /// @{
    __GPU__ void _conv();                   ///< init convolution layer
    __GPU__ void _loss(t4_loss op);         ///< calculate loss
    __GPU__ void _sgd();
    __GPU__ void _adam();
    /// @}
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
