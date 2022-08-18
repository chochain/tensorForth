/**
 * @file
 * @brief tensorForth - TensorVM, extended ForthVM classes, to handle tensor ops
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENVM_H
#define TEN4_SRC_TENVM_H
#include "eforth.h"                         /// extending ForthVM

#define EXP(d)    (expf(d))                 /**< exponential(float) */
#define TTOS      (mmu.du2ten(top))         /**< tensor on TOS      */
#define TNOS      (mmu.du2ten(ss[-1]))      /**< tensor on NOS      */
#define TOS1T     (IS_OBJ(top) && TTOS.is_tensor())
#define TOS2T     (TOS1T && IS_OBJ(ss[-1]) && TNOS.is_tensor())
#define TOS3T     (TOS2T && IS_OBJ(ss[-2]) && mmu.du2ten(ss[-2]).is_tensor())

typedef enum {
    KEEP = false,
    DROP = true
} t4_drop_opt;

class TensorVM : public ForthVM {
public:
#if   !T4_ENABLE_OBJ
    __GPU__ TensorVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        ForthVM(khz, istr, ostr, mmu0) {}
    __GPU__ void init() { ForthVM::init(); }
    
#else // T4_ENABLE_OBJ
    __GPU__ TensorVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        ForthVM(khz, istr, ostr, mmu0) {
        VLOG1("\\  ::TensorVM(...) sizeof(Tensor)=%ld\n", sizeof(Tensor));
    }
    __GPU__ void virtual init();            ///< override ForthVM
    
protected:
    int    ten_lvl = 0;                     ///< tensor input level
    int    ten_off = 0;                     ///< tensor offset (storage index)
    
    ///
    /// override literal handler
    ///
    __GPU__ void tprint(DU d);              ///< tensor dot (print)
    __GPU__ int  number(char *str) final;   ///< TODO: CC - this worked, why?
    ///
    /// tensor ops based on number of operands
    ///
    __GPU__ void xop1(t4_ten_op op, DU v=DU0);                   /// 1-operand ops in-place
    __GPU__ void xop1x(t4_ten_op op);                            /// 1-operand ops with new tensor
    __GPU__ void xop2(t4_ten_op op, t4_drop_opt x=KEEP);         /// 2-operand ops
    
private:
    ///
    /// tensor ops based on data types
    ///
    __GPU__ void _ss_op(t4_ten_op op);                           ///< scalar-scalar (Forth) ops
    __GPU__ void _ts_op(t4_ten_op op, t4_drop_opt x, bool swap); ///< tensor-scalar broadcast op
    __GPU__ void _tt_op(t4_ten_op op, t4_drop_opt x);            ///< tensor-tensor ops
    ///
    /// tensor-tensor ops
    ///
    __GPU__ Tensor &_tinv(Tensor &A);                            ///< matrix inversion
    __GPU__ Tensor *_tdot(Tensor &A, Tensor &B, bool *tt);       ///< matrix-matrix multiplication @
    __GPU__ Tensor *_tdiv(Tensor &A, Tensor &B);                 ///< matrix-matrix division (no broadcast)
    __GPU__ Tensor *_solv(Tensor &A, Tensor &B);                 ///< solve linear equation Ax = b
    __GPU__ void   _gemm();                                      ///< GEMM C' = alpha * A x B + beta * C
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_TENVM_H
