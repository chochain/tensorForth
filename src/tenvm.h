/**
 * @file
 * @brief tensorForth - TensorVM, extended ForthVM classes, to handle tensor ops
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENVM_H
#define TEN4_SRC_TENVM_H
#include "eforth.h"                         /// extending ForthVM

class TensorVM : public ForthVM {
public:
    __GPU__ TensorVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu)
        : ForthVM(khz, istr, ostr, mmu) {
        INFO("\\  ::TensorVM(...) sizeof(Tensor)=%ld\n", sizeof(Tensor));
    }
    __GPU__ void init_t();                  ///< TODO: CC - polymorphism does not work here?
    __GPU__ void init() final { init_t(); } ///< so fake it

protected:
    int   ten_lvl = 0;                      ///< tensor input level
    int   ten_off = 0;                      ///< tensor offset (array index)
    ///
    /// override literal handler
    ///
    __GPU__ int  number(char *str) final;   ///< TODO: CC - this worked, why?
    ///
    /// mmu proxy functions
    ///
    __GPU__ void add_tensor(DU n);          ///< add tensor to parameter field
    ///
    /// tensor ops
    ///
    __GPU__ DU   tadd(bool sub=false);      ///< matrix-matrix addition (or subtraction)
    __GPU__ DU   tmul();                    ///< matrix multiplication (no broadcast)
    __GPU__ DU   tdiv();                    ///< matrix division (no broadcast)
    __GPU__ DU   tinv();                    ///< TODO: matrix inverse (Gaussian Elim.?)
    __GPU__ DU   ttrans();                  ///< matrix transpose
    __GPU__ void gemm();                    ///< GEMM C' = alpha * A x B + beta * C
};
#endif // TEN4_SRC_TENVM_H
