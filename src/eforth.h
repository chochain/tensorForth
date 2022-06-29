/**
 * @file
 * @brief tensorForth - eForth core classes
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_EFORTH_H
#define TEN4_SRC_EFORTH_H
#include "vector.h"         // Forth vector (includes util.h)
#include "tensor.h"
#include "aio.h"            // Forth async IO (includes Istream, Ostream)
///
///@name Cross platform support
///@{
#define ENDL            '\n'
#define yield()                              /** TODO: multi-VM */
#define delay(ticks)    { clock_t t = clock()+ticks; while (clock()<t) yield(); }
///@}
///
/// Forth virtual machine class
///
typedef enum { VM_READY=0, VM_RUN, VM_WAIT, VM_STOP } vm_status;
///
/// Forth Virtual Machine
///
class ForthVM {
public:
    vm_status status = VM_READY;            ///< VM status
    DU        top    = DU0;                 ///< cached top of stack
    Vector<DU,   T4_RS_SZ> rs;              ///< return stack
    Vector<DU,   0>        ss;              ///< parameter stack (setup in ten4.cu)

    __GPU__ ForthVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu);
    __GPU__ void init();
    __GPU__ void outer();

private:
    Istream       &fin;                     ///< VM stream input
    Ostream       &fout;                    ///< VM stream output
    MMU           &mmu;                     ///< memory managing unit
    Code          *dict;                    ///< dictionary array

    int   khz;                              ///< VM clock rate
    U32   *ptop   = (U32*)&top;             ///< 32-bit mask for top
    
    bool  ucase   = true;                   ///< case insensitive
    bool  compile = false;                  ///< compiling flag
    int   radix   = 10;                     ///< numeric radix
    IU    WP      = 0;                      ///< word and parameter pointers
    IU    IP      = 0;                      ///< instruction pointer

    char  idiom[T4_STRBUF_SZ];              ///< terminal input buffer
    int   ten_lvl = 0;                      ///< tensor input level
    int   ten_off = 0;                      ///< tensor offset (array index)

    __GPU__ __INLINE__ DU POP()             { DU n=top; top=ss.pop(); return n; }
    __GPU__ __INLINE__ void PUSH(DU v)      { ss.push(top); top = v; }
    __GPU__ __INLINE__ void PUSH(Tensor &t) { ss.push(top); top = mmu.ten2du(t); }

    __GPU__ int  find(const char *s);       ///< search dictionary reversely
    ///
    /// Forth inner interpreter
    ///
    __GPU__ char *next_idiom();
    __GPU__ char *scan(char c);
    __GPU__ void nest();
    ///
    /// compiler proxy funtions to reduce verbosity
    ///
    __GPU__ void add_w(IU w);               ///< append a word pfa to pmem
    __GPU__ void add_iu(IU i);              ///< append an instruction unit to parameter memory
    __GPU__ void add_du(DU d);              ///< append a data unit to pmem
    __GPU__ void add_str(const char *s);    ///< append a string to pmem
    __GPU__ void add_tensor(DU d);          ///< append a literal into tensor storage
    __GPU__ void call(IU w);                ///< execute word by index
    ///
    /// tensor methods
    ///
    __GPU__ DU   tadd(bool sub=false);      ///< matrix-matrix addition (or subtraction)
    __GPU__ DU   tmul();                    ///< matrix-matrix multiplication (no broadcast)
    __GPU__ DU   tinv();                    ///< TODO: matrix inverse (Gaussian Elim.?)
    __GPU__ void gemm();                    ///< GEMM C' = alpha * A x B + beta * C
    ///
    /// output methods
    ///
    __GPU__ void dot(DU v);
    __GPU__ void dot_r(int n, DU v);
    __GPU__ void ss_dump(int n);
};
#endif // TEN4_SRC_EFORTH_H
