/**
 * @file
 * @brief tensorForth - TensorVM, extended ForthVM classes, to handle tensor ops
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENVM_H
#define TEN4_SRC_TENVM_H
#include "eforth.h"                         /// extending ForthVM

#define SCALAR(v) (*(U8*)&(v) &= ~T4_OBJ_FLAG)  /**< tensor flag mask for top       */
#define EXP(d)    (expf(d))                     /**< exponential(float)             */
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
    /// mmu proxy functions
    ///
    __GPU__ void add_to_tensor(DU n);       ///< add tensor to parameter field
    ///
    /// tensor ops
    ///
    __GPU__ void ssop(t4_mat_op op);        ///< scalar-scalar (Forth) ops
    __GPU__ void tsop(t4_mat_op op, t4_drop_opt x, bool swap); ///< tensor-scalar broadcast op
    __GPU__ void tmat(t4_mat_op op, t4_drop_opt x); ///< matrix-matrix element ops (Hadamard)
    __GPU__ void tmul(t4_drop_opt x);       ///< matrix-matrix multiplication @
    __GPU__ void tdiv(t4_drop_opt x);       ///< matrix-matrix division (no broadcast)
    __GPU__ void tinv();                    ///< matrix inversion (Gauss-Jordan)
    __GPU__ void tlu();                     ///< matrix LU decomposition
    __GPU__ void tdet();                    ///< matrix determinant (via LU)
    __GPU__ void ttrans();                  ///< matrix transpose
    __GPU__ void solve();                   ///< solve linear equation Ax = b
    __GPU__ void gemm();                    ///< GEMM C' = alpha * A x B + beta * C
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_TENVM_H
