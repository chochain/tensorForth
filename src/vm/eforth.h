/**
 * @file
 * @brief tensorForth - eForth core classes
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_EFORTH_H
#define TEN4_SRC_EFORTH_H
#include "vm.h"             // VM base class in ../vm
///
///@name Data conversion
///@{
#define POPi         (INT(POP()))                  /**< convert popped DU as an IU     */
#define FIND(s)      (mmu.find(s, compile, ucase)) /**< find input idiom in dictionary */
///@}
///
/// Forth Virtual Machine
///
class ForthVM : public VM {
public:
    __GPU__ ForthVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0)
        : VM(khz, istr, ostr, mmu0), dict(mmu0->dict()) {
        VLOG1("\\  ::ForthVM[%d](dict=%p) sizeof(Code)=%ld\n",
              vid, dict, sizeof(Code));
    }
    __GPU__ virtual void init();            ///< override VM
    
protected:
    Code  *dict;                            ///< dictionary array
    Vector<DU, T4_RS_SZ> rs;                ///< return stack
    
    int   radix   = 10;                     ///< numeric radix
    bool  ucase   = true;                   ///< case insensitive
    IU    WP      = 0;                      ///< word and parameter pointers
    IU    IP      = 0;                      ///< instruction pointer
    IU    RS      = 0;                      ///< call stack depth
    ///
    /// stack short hands
    ///
    __GPU__ __INLINE__ DU POP()           { DU n=top; top=ss.pop(); return n; }
    __GPU__ __INLINE__ DU PUSH(DU v)      { ss.push(top); return top = v; }
 #if T4_ENABLE_OBJ
    __GPU__ __INLINE__ DU PUSH(T4Base &t) { ss.push(top); return top = mmu.obj2du(t); }
#endif // T4_ENABLE_OBJ
    ///
    /// Forth outer interpreter
    ///
    __GPU__ virtual int resume();           ///< resume suspended work
    __GPU__ virtual int parse(char *str);   ///< parse command string
    __GPU__ virtual int number(char *str);  ///< parse input as number
    ///
    /// Forth inner interpreter
    ///
    __GPU__ void nest();                    ///< inner interpreter
    __GPU__ void call(IU w);                ///< execute word by index
    ///
    /// compiler proxy funtions to reduce verbosity
    ///
    __GPU__ void add_w(IU w);               ///< append a word pfa to pmem
    __GPU__ void add_iu(IU i);              ///< append an instruction unit to parameter memory
    __GPU__ void add_du(DU d);              ///< append a data unit to pmem
    __GPU__ void add_str(const char *s);    ///< append a string to pmem
};
#endif // TEN4_SRC_EFORTH_H
