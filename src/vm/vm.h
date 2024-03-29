/**
 * @file
 * @brief VM class - eForth VM virtual class interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_VM_H
#define TEN4_SRC_VM_H
#include "aio.h"            // async IO (includes Istream, Ostream), in ../io
///
///@name Cross platform support
///@{
#define ENDL         '\n'
#define delay(ticks) { U64 t = clock64() + (ticks * mmu.khz()); while ((U64)clock64()<t) yield(); }
#define yield()                        /**< TODO: multi-VM  */
///@}
#define VLOG1(...)         if (mmu.trace() > 0) INFO(__VA_ARGS__);
#define VLOG2(...)         if (mmu.trace() > 1) INFO(__VA_ARGS__);
///
/// virtual machine base class
///
typedef enum { VM_READY=0, VM_RUN, VM_WAIT, VM_STOP } vm_state;
class VM {
public:
    int       vid    = 0;              ///< VM id
    vm_state  state  = VM_READY;       ///< VM state
    DU        top    = DU0;            ///< cached top of stack
    Vector<DU, 0> ss;                  ///< parameter stack (setup in ten4.cu)

    __GPU__ VM(int id, Istream *istr, Ostream *ostr, MMU *mmu);

    __GPU__ virtual void init() { VLOG1("VM::init ok\n"); }
    __GPU__ virtual void outer();

protected:
    Istream  &fin;                     ///< VM stream input
    Ostream  &fout;                    ///< VM stream output
    MMU      &mmu;                     ///< memory managing unit

    U32   *ptop   = (U32*)&top;        ///< 32-bit mask for top
    bool  compile = false;             ///< compiling flag

    char  idiom[T4_STRBUF_SZ];         ///< terminal input buffer
    ///
    /// inner interpreter handlers
    ///
    __GPU__ virtual int resume()          { return 0; }
    __GPU__ virtual int pre(char *str)    { return 0; }
    __GPU__ virtual int parse(char *str)  { return 0; }
    __GPU__ virtual int number(char *str) { return 0; }
    __GPU__ virtual int post()            { return 0; }
    ///
    /// input stream handler
    ///
    __GPU__ char *next_idiom()      { fin >> idiom; return idiom; }
    __GPU__ char *scan(char delim)  { fin.get_idiom(idiom, delim); return idiom; }
    ///
    /// output methods
    ///
    __GPU__ void dot(DU v)          { fout << " " << v; }
    __GPU__ void dot_r(int n, DU v) { fout << setw(n) << v; }
    __GPU__ void ss_dump(int n=0)   {
        ss[T4_SS_SZ-1] = top;        /// * put top at the tail of ss (for host display)
        fout << opx(OP_SS, n ? n : ss.idx);
    }
};
#endif // TEN4_SRC_VM_H
