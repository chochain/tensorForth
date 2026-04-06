/**
 * @file
 * @brief TensorVM class , extended ForthVM classes, tensor handler interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (!defined(__VM_TENVM_H) && T4_DO_OBJ)
#define __VM_TENVM_H
#include "eforth.h"                         /// extending ForthVM

namespace t4::vm {

#define VOP(...)  static const char *_op[] = { __VA_ARGS__ }
#define VLOG(...) if (sys.trace()) { INFO(__VA_ARGS__); }
///
///@name multi-dispatch checker macros
///@{
#define TTOS      ((Tensor&)mmu.du2obj(tos))        /**< tensor on TOS      */
#define TNOS      ((Tensor&)mmu.du2obj(ss[-1]))     /**< tensor on NOS      */
#define TOS1T     (IS_OBJ(tos) && TTOS.is_tensor())
#define TOS2T     (ss.idx > 0 && TOS1T && IS_OBJ(ss[-1]) && TNOS.is_tensor())
#define TOS3T     (ss.idx > 1 && TOS2T && IS_OBJ(ss[-2]) && mmu.du2obj(ss[-2]).is_tensor())
///@}
///@name Tensor (multi-dimension array) class
///@{
class TensorVM : public ForthVM {
    using Tensor = mu::Tensor;             ///< alias
public:
    __HOST__ TensorVM(int id, System &sys) : ForthVM(id, sys) {
        TRACE("\\    ::TensorVM[%d]\n", id);
    }
    __HOST__ virtual void init();           ///< override ForthVM.init()
    
protected:
    U32    ten_off = 0;                     ///< tensor offset (storage index)
    int    ten_lvl = 0;                     ///< tensor input level
    ///
    /// override literal handler
    ///
    __HOST__ virtual int process(char *str); ///< TODO: CC - worked without 'final', why?
    ///
    /// stack operator short hands (override eforth.h)
    ///
    __HOST__ __INLINE__ void FREE(Tensor &t)  { mmu.free(t); }
    __HOST__ __INLINE__ DU   PUSH(T4Base &t)  { ss.push(tos); return tos = mmu.obj2du(t); }
    __HOST__ __INLINE__ DU   PUSH(DU v)       { ss.push(tos); return tos = v; }
    __HOST__ __INLINE__ DU   COPY(DU d) {                 ///< hard copy
        return (IS_OBJ(d))
            ? mmu.obj2du(COPY((Tensor&)mmu.du2obj(d)))
            : d;
    }
    __HOST__ __INLINE__ Tensor &COPY(Tensor &t) { return mmu.copy(t); }
    ///
    /// tensor ops based on number of operands
    ///
    __HOST__ void xop1(math_op op, DU v=DU0);                ///< 1-operand ops in-place
    __HOST__ void xop2(math_op op, t4_drop_opt x=T_KEEP);    ///< 2-operand ops
    __HOST__ void blas1(t4_ten_op op);                       ///< 1-operand ops with new tensor
    __HOST__ void blas2(t4_ten_op op, t4_drop_opt x=T_KEEP); ///< 2-operand tensor ops
    __HOST__ void gemm();                                    ///< GEMM C' = alpha * A x B + beta * C
    
private:
    ///
    /// tensor ops based on data types
    ///
    __HOST__ void   _ss_op(math_op op);                      ///< scalar-scalar (eForth) ops
    __HOST__ Tensor &_st_op(math_op op, t4_drop_opt x);      ///< scalar tensor op (broadcast)
    __HOST__ Tensor &_ts_op(math_op op, t4_drop_opt x);      ///< tensor scalar op (broadcast)
    __HOST__ Tensor &_tt_op(math_op op);                     ///< tensor tensor op
    ///
    /// tensor-tensor ops
    ///
    __HOST__ Tensor &_tinv(Tensor &A);                       ///< matrix inversion
    __HOST__ Tensor &_tdot(Tensor &A, Tensor &B);            ///< matrix-matrix multiplication @
    __HOST__ Tensor &_tdiv(Tensor &A, Tensor &B);            ///< matrix-matrix division (no broadcast)
    __HOST__ Tensor &_solv(Tensor &A, Tensor &B);            ///< solve linear equation Ax = b
    ///
    /// tensor IO
    ///
    __HOST__ void   _pickle(bool save);                      ///< save/load a tensor to/from a file
    __HOST__ void   _ttos_dump();
};
///@}

} // namespace t4::vm

#endif // (!defined(__VM_TENVM_H) && T4_DO_OBJ)
