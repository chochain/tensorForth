/**
 * @file
 * @brief ForthVM class - eForth VM classes interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_VM_EFORTH_H
#define TEN4_SRC_VM_EFORTH_H
#include "vm.h"                         ///< VM base class in ../vm
#include "param.h"                      ///< Parameter field
///
/// Forth Virtual Machine operational macros to reduce verbosity
/// Note:
///    also we can change pmem implementation anytime without affecting opcodes defined below
///
///@name parameter memory load/store macros
///@{
#define PFA(w)    (dict[(IU)(w)].pfa)                 /**< PFA of given word id                 */
#define HERE      (mmu.here())                       /**< current context                      */
#define MEM(a)    (mmu.pmem((IU)(a)))                /**< parameter memory by offset address   */
#define CELL(a)   (*(DU*)MEM(a))                      /**< fetch a cell from parameter memory   */
#define LAST      (mmu.dict(mmu.dict._didx-1))      /**< last colon word defined              */
#define BASE      ((U8*)MEM(base))                    /**< pointer to user area per VM          */
#define SETJMP(a) (((Param*)MEM(a))->ioff = HERE)     /**< set branch target                    */
#define SS2I      ((id<<10)|(ss.idx>=0 ? ss.idx : 0)) /**< ss_dump parameter (composite)        */
#define POPi      (D2I(POP()))
///@}
///@name progress status macros
///@{
#define VM_HDR(fmt, ...)                     \
    DEBUG("\e[%dm[%02d.%d]%-4x" fmt "\e[0m", \
          (id&7) ? 38-(id&7) : 37, id, state, ip, ##__VA_ARGS__)
#define VM_TLR(fmt, ...)                     \
    DEBUG("\e[%dm" fmt "\e[0m\n",            \
          (id&7) ? 38-(id&7) : 37, ##__VA_ARGS__)
#define VM_LOG(fmt, ...)                     \
    VM_HDR(fmt, ##__VA_ARGS__);              \
    DEBUG("\n")
///@}
///@name Dictionary Compiler macros
///@note - a lambda without capture can degenerate into a function pointer
///@{
#define ADD_CODE(n, g, im) {           \
    auto f = [this] __GPU__ (){ g; };  \
    mmu.add_word(n, f, im);           \
}
#define CODE(n, g) ADD_CODE(n, g, false)
#define IMMD(n, g) ADD_CODE(n, g, true)
///@}
///@name Forth Virtual Machine class
///@{
class ForthVM : public VM {
public:
    __GPU__ ForthVM(int id, System &sys);
    
    __GPU__ virtual void init();      ///< override VM
    
protected:
    IU    ip     = 0;                 ///< instruction pointer
    DU    tos    = -DU1;              ///< cached top of stack
    
    bool  compile= false;
    IU    base   = 0;
    
    Code  *dict  = 0;                 ///< dictionary array (cached)
    U32   *ptos  = (U32*)&tos;        ///< 32-bit mask for tos
    ///
    /// Forth outer interpreter
    ///
    __GPU__ virtual int resume();             ///< resume suspended work
    __GPU__ virtual int process(char *idiom); ///< process command string
    __GPU__ virtual int post();               ///< for tracing
    ///
    /// outer interpreter
    ///
    __GPU__ IU parse(char *idiom);            ///< parse command string
    __GPU__ DU number(char *idiom, char **p); ///< parse input as number
    ///
    /// Forth inner interpreter
    ///
    __GPU__ void nest();                      ///< inner interpreter
    __GPU__ void call(IU w);                  ///< execute word by index
    ///
    /// stack operator short hands
    ///
    __GPU__ __INLINE__ IU  FIND(char *name) { return mmu.find(name);  }
    __GPU__ __INLINE__ DU  POP()            { DU n=tos; tos=ss.pop(); return n; }
    __GPU__ __INLINE__ DU  PUSH(DU v)       { ss.push(tos); return tos = v;     }
    ///
    /// Dictionary compiler proxy macros to reduce verbosity
    ///
    __GPU__ __INLINE__ void add_iu(IU i)   { mmu.add((U8*)&i, sizeof(IU)); }
    __GPU__ __INLINE__ void add_du(DU d)   { mmu.add((U8*)&d, sizeof(DU)); }
    __GPU__ __INLINE__ void add_w(Param p) { add_iu(p.pack); }
    __GPU__ void add_w(IU w) {                ///< compile a word index into pmem
        Code &c = dict[w];
        IU   ix = c.udf ? c.pfa : mmu.XTOFF(c.xt);
        DEBUG(" add_w(%d) => ioff=%x %s\n", w, ix, c.name);
        Param p(MAX_OP, ix, c.udf);
        add_w(p);
    }
    __GPU__ int  add_str(const char *s, bool adv=true) {
        int sz = STRLENB(s)+1;                ///< calculate string length
        sz = ALIGN(sz);                       /// * then adjust alignment (combine?)
        mmu.add((U8*)s, sz, adv);
        return sz;
    }
    __GPU__ void add_p(                       ///< add primitive word
        prim_op op, IU ix=0, bool u=false, bool exit=false) {
        Param p(op, ix, u, exit);
        add_w(p);
    };
    __GPU__ void add_lit(DU v, bool exit=false) {  ///< add a literal/varirable
        add_p(LIT, 0, false, exit);
        add_du(v);                            /// * store in extended IU
    }
    
private:    
    ///
    /// compiler helpers
    ///
    __GPU__ int  _def_word();                 ///< define a new word
    __GPU__ void _forget();                   ///< clear dictionary
    __GPU__ void _quote(prim_op op);          ///< string helper
    __GPU__ void _to_value();                 ///< update a constant/value
    __GPU__ void _is_alias();                 ///< create alias function
};
///@}
#endif // TEN4_SRC_VM_EFORTH_H
