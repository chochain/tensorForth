/**
 * @file
 * @brief NetVM class - extend TensorVM class, Neural Network VM interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (!defined(__VM_NETVM_H) && T4_DO_OBJ && T4_DO_NN)
#define __VM_NETVM_H
#include "tenvm.h"                // extending TensorVM
#include "nn/model.h"

#define MTOS     ((Model&)mmu.du2obj(tos))                                       /** Network Model on TOS   */
#define MNOS     ((Model&)mmu.du2obj(ss[-1]))                                    /** Network Model on NOS   */
#define TOS1D    (IS_OBJ(tos) && (TTOS.is_tensor() || TTOS.is_dataset()))
#define RS1D     (IS_OBJ(rs[-1]) && mmu.du2obj(rs[-1]).is_dataset())
#define IS_M(v)  (IS_OBJ(v) && mmu.du2obj(v).is_model())                         /** check param is a model */
#define M1V      (ss.idx > 0 && !IS_OBJ(tos) && IS_M(ss[-1]))                    /** NOS model w 1-param    */
#define M2V      (ss.idx > 1 && !IS_OBJ(tos) && !IS_OBJ(ss[-1]) && IS_M(ss[-2])) /** ss[-2] model w 2-param */
#define MTV      (ss.idx > 1 && !IS_OBJ(tos) && IS_OBJ(ss[-1]) && IS_M(ss[-2]))  /** ss[-2] model tensor w 1-param */

class NetVM : public TensorVM {
public:
    __GPU__ NetVM(int id, System &sys) : TensorVM(id, sys) {
        TRACE("\\      ::NetVM[%d]\n", id);
    }
    ///
    /// @name Class Object Constructor
    /// @{
    __GPU__ virtual void init() final;           ///< override TensorVM, TODO: does not work without 'final', why?
    
    __GPU__ void predict(Tensor &I, Tensor &P);  ///< predict result

private:
    ///
    /// @name model and dataset ops
    /// @{
    __GPU__ void _nnop(t4_layer op);
    __GPU__ void _pickle(bool save);             ///< override TenVM::_pickle
    __GPU__ void _get_parm(int n);               ///< fetch tensor parameters n=0:W, 1:B, 2:dW, 3:dB
    __GPU__ void _set_parm(int n);               ///< set tensor parameters (for debugging)
    /// @}
    /// @name Convolution, loss and Gradiant ops
    /// @{
    __GPU__ void _conv(U16 k=3);                 ///< init convolution layer
    __GPU__ void _loss(t4_loss op);              ///< calculate loss
    /// @}
};

#endif // (!defined(__VM_NETVM_H) && T4_DO_OBJ && T4_DO_NN)
