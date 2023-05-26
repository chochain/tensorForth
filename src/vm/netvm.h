/**
 * @file
 * @brief NetVM class - extend TensorVM class, Neural Network VM interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_NETVM_H
#define TEN4_SRC_NETVM_H
#include "model.h"                // in ../mmu
#include "tenvm.h"                // extending TensorVM

#define MTOS     ((Model&)mmu.du2obj(top))                         /** Network Model on TOS   */
#define MNOS     ((Model&)mmu.du2obj(ss[-1]))                      /** Network Model on NOS   */
#define TOS1D    (IS_OBJ(top) && (TTOS.is_tensor() || TTOS.is_dataset()))
#define RS1D     (IS_OBJ(rs[-1]) && mmu.du2obj(rs[-1]).is_dataset())
#define IS_M(v)  (IS_OBJ(v) && mmu.du2obj(v).is_model())           /** check param is a model */
#define M1V      (!IS_OBJ(top) && IS_M(ss[-1]))                    /** NOS model w 1-param    */
#define M2V      (!IS_OBJ(top) && !IS_OBJ(ss[-1]) && IS_M(ss[-2])) /** ss[-2] model w 2-param */

class NetVM : public TensorVM {
public:
#if   !T4_ENABLE_OBJ
    __GPU__ NetVM(Istream *istr, Ostream *ostr, MMU *mmu0) : TensorVM(istr, ostr, mmu0) {}
    __GPU__ virtual void init() { TensorVM::init(); }

#else // T4_ENABLE_OBJ
    ///
    /// @name Class Object Constructor
    /// @{
    __GPU__ NetVM(Istream *istr, Ostream *ostr, MMU *mmu0) : TensorVM(istr, ostr, mmu0) {
        VLOG1("\\  ::NetVM[%d](...) sizeof(Model)=%d\n", vid, (int)sizeof(Model));
    }
    __GPU__ virtual void init() final; ///< override TensorVM, TODO: does not work without 'final', why?
    
    __GPU__ void nnop(t4_layer op);
    __GPU__ void predict(Tensor &I, Tensor &P);  ///< predict result

private:
    /// @name model and dataset ops
    /// @{
    __GPU__ void _pickle(bool save);             ///<
    __GPU__ void _fetch(DU d, bool more);        ///< fetch or rewind dataset
    /// @}
    /// @name Convolution, loss and Gradiant ops
    /// @{
    __GPU__ void _conv(U16 k=3);                 ///< init convolution layer
    __GPU__ void _loss(t4_loss op);              ///< calculate loss
    /// @}
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_NETVM_H
