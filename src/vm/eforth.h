/**
 * @file
 * @brief ForthVM class - eForth VM classes interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_EFORTH_H
#define TEN4_SRC_EFORTH_H
#include "vm.h"                         ///< VM base class in ../vm

class Param {
    union {
        U32 xtoff;
        struct {
            U16  usr  : 1;    ///> user defined words
            U16  prim : 1;    ///> primitive
            U16  xx1  : 14;   ///> reserved
            U16  didx;        ///> offset or index
        };
    };
    Param(IU off, bool u=false, bool p=false) : xtoff(off) { usr=u; prim=p; }
};
///
/// macros for microcode construction
///
#define ADD_CODE(n, g, im) {            \
    auto f = [this] __GPU__ (){ g; };   \
    mmu->add_word(n, f, im);            \
}
#define CODE(n, g)  ADD_CODE(n, g, false)
#define IMMD(n, g)  ADD_CODE(n, g, true)
///
///@name Data conversion
///@{
#define POPi    (INT(POP()))          /**< convert popped DU as an IU     */
///@}
///
/// Forth Virtual Machine
///
class ForthVM : public VM {
public:
    __GPU__ ForthVM(int id, System *sys);
    
    __GPU__ virtual void init();      ///< override VM
    
protected:
    IU    WP     = 0;                 ///< word pointer
    IU    IP     = 0;                 ///< instruction pointer
    DU    tos    = DU0;               ///< cached top of stack
    
    Code  *dict  = 0;                 ///< dictionary array (cached)
    U8    *base  = 0;                 ///< radix (base)
    U32   *ptos  = (U32*)&tos;        ///< 32-bit mask for tos
    ///
    /// stack short hands
    ///
    __GPU__ __INLINE__ int FIND(char *name) { return mmu->find(name, compile);  }
    __GPU__ __INLINE__ DU  POP()            { DU n=tos; tos=ss.pop(); return n; }
    __GPU__ __INLINE__ DU  PUSH(DU v)       { ss.push(tos); return tos = v;     }
#if T4_ENABLE_OBJ    
    __GPU__ __INLINE__ DU  PUSH(T4Base &t)  { ss.push(tos); return tos = T4Base::obj2du(t); }
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
    /// Dictionary compiler proxy macros to reduce verbosity
    ///
    __GPU__ __INLINE__ void add_w(IU w)  {  ///< compile a word into pmem
        add_iu(w);
        DEBUG(" add_w(%d) => %s\n", w, dict[w].name);
    }
//    __GPU__ __INLINE__ void add_w(Param p) { add_w((IU)p.pfa); }
    __GPU__ __INLINE__ void add_iu(IU i) { mmu->add((U8*)&i, sizeof(IU)); }
    __GPU__ __INLINE__ void add_du(DU d) { mmu->add((U8*)&d, sizeof(DU)); }
    __GPU__ __INLINE__ void add_str(const char *s, bool adv=true) {
        int sz = STRLENB(s)+1; sz = ALIGN(sz);  ///> calculate string length, then adjust alignment (combine?)
        mmu->add((U8*)s, sz, adv);
    }
};
#endif // TEN4_SRC_EFORTH_H
