/**
 * @file
 * @brief tensorForth - TensorVM, extended ForthVM classes, to handle tensor ops
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENVM_H
#define TEN4_SRC_TENVM_H
#include "eforth.h"                         /// extending ForthVM

#define NO_OBJ(v) (*(U8*)&(v) &= ~T4_OBJ_FLAG)  /**< tensor flag mask for top       */
#define EXP(d)    (expf(d))                     /**< exponential(float)             */

class TensorVM : public ForthVM {
public:
#if   !T4_ENABLE_OBJ
    __GPU__ TensorVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        ForthVM(khz, istr, ostr, mmu0) {}
    __GPU__ void init_t() { ForthVM::init_f(); }
    
#else // T4_ENABLE_OBJ
    __GPU__ TensorVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0) :
        ForthVM(khz, istr, ostr, mmu0) {
        VLOG1("\\  ::TensorVM(...) sizeof(Tensor)=%ld\n", sizeof(Tensor));
    }
    __GPU__ void init() final { init_t(); } ///< TODO: CC - polymorphism does not work here?
    __GPU__ void init_t();                  ///< so fake it
    
protected:
    int   ten_lvl = 0;                      ///< tensor input level
    int   ten_off = 0;                      ///< tensor offset (storage index)
    ///
    /// override literal handler
    ///
    __GPU__ void tprint(DU d);              ///< tensor dot (print)
    __GPU__ int  number(char *str) final;   ///< TODO: CC - this worked, why?
    ///
    /// mmu proxy functions
    ///
    __GPU__ void add_to_tensor(DU n);      ///< add tensor to parameter field
    ///
    /// tensor ops
    ///
    __GPU__ void texp();                    ///< element-wise all tensor elements
    __GPU__ void tadd(bool sub=false);      ///< matrix-matrix addition (or subtraction)
    __GPU__ void tmul();                    ///< matrix multiplication (no broadcast)
    __GPU__ void tdiv();                    ///< matrix division (no broadcast)
    __GPU__ void tinv();                    ///< matrix inversion (Gauss-Jordan)
    __GPU__ void tlu();                     ///< matrix LU decomposition
    __GPU__ void tluinv();                  ///< inversion of a LU matrix
    __GPU__ void tdet();                    ///< matrix determinant (via LU)
    __GPU__ void ttrans();                  ///< matrix transpose
    __GPU__ void gemm();                    ///< GEMM C' = alpha * A x B + beta * C
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_TENVM_H
