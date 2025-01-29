/**
 * @file
 * @brief ForthVM class - eForth VM classes interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_EFORTH_H
#define TEN4_SRC_EFORTH_H
#include "vm.h"                         ///< VM base class in ../vm
///
///@name Data conversion
///@{
#define POPi    (INT(POP()))                   /**< convert popped DU as an IU     */
#define FIND(s) (sys->mu->find(s, compile))    /**< find input idiom in dictionary */
///@}
///
/// Forth Virtual Machine
///
class ForthVM : public VM {
public:
    __HOST__ ForthVM(int id, System *sys)
        : VM(id, sys), dict(sys->mu->dict()) {
        base = sys->mu->pmem(id);
        VLOG1("\\  ::ForthVM[%d](dict=%p) sizeof(Code)=%ld\n", id, dict, sizeof(Code));
    }
    __GPU__ virtual void init();      ///< override VM
    
protected:
    Code      *dict;                  ///< dictionary array (cached)
    
    IU        WP     = 0;             ///< word pointer
    IU        IP     = 0;             ///< instruction pointer
    DU        tos    = DU0;           ///< cached top of stack
    
    Vector<DU, T4_RS_SZ> rs;          ///< return stack
    
    U32   *ptos   = (U32*)&tos;       ///< 32-bit mask for top
    U8    *base   = 0;                ///< radix (base)
    ///
    /// stack short hands
    ///
    __GPU__ __INLINE__ DU POP()           { DU n=tos; tos=ss.pop(); return n; }
    __GPU__ __INLINE__ DU PUSH(DU v)      { ss.push(tos); return tos = v; }
#if T4_ENABLE_OBJ    
    __GPU__ __INLINE__ DU PUSH(T4Base &t) { ss.push(tos); return tos = mmu.obj2du(t); }
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
    __GPU__ void add_str(const char *s, bool adv=true); ///< append a string to pmem
};
#endif // TEN4_SRC_EFORTH_H
