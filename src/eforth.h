/**
 * @file
 * @brief tensorForth - eForth core classes
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_EFORTH_H
#define TEN4_SRC_EFORTH_H
#include "vm.h"             // VM base class
///
///@name Cross platform floating-point support
///@{
#define MOD(t,n)     (fmod(t, n))                  /**< fmod two floating points       */
#define ZERO(d)      (ABS(d) < DU_EPS)             /**< zero check                     */
#define BOOL(f)      (ZERO(f) ? -1 : 0)            /**< default boolean representation */
#define ABS(d)       (fabs(d))                     /**< absolute value                 */
#define EXP(v)       (expf(v))                     /**< exponential(float)             */
///@}
///@name Data conversion
///@{
#define INT(f)       (static_cast<int>(f + 0.5))   /**< cast float to int              */
#define I2D(i)       (static_cast<DU>(i))          /**< cast int back to float         */
#define DU_ONLY(v)   (*(U32*)&(v) &= ~1)           /**< tensor flag mask for top       */
#define POPi         (INT(POP()))                  /**< convert popped DU as an IU     */
#define FIND(s)      (mmu.find(s, compile, ucase)) /**< find input idiom in dictionary */
///@}
///
/// Forth Virtual Machine
///
class ForthVM : public VM {
public:
    __GPU__ ForthVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu)
        : VM(khz, istr, ostr, mmu), dict(mmu->dict()) {
        if (mmu->trace() > 0) {
            printf("\\  ::ForthVM[%d](dict=%p) sizeof(Code)=%ld\n", blockIdx.x, dict, sizeof(Code));
        }
    }
    __GPU__ void virtual init() override { init_f(); } ///< TODO: CC - polymorphism does not work in kernel?
    __GPU__ void init_f();                             ///< so fake it for now
    
protected:
    Code  *dict;                            ///< dictionary array
    Vector<DU, T4_RS_SZ> rs;                ///< return stack
    
    bool  ucase   = true;                   ///< case insensitive
    int   radix   = 10;                     ///< numeric radix
    IU    WP      = 0;                      ///< word and parameter pointers
    IU    IP      = 0;                      ///< instruction pointer
    ///
    /// stack short hands
    ///
    __GPU__ __INLINE__ DU POP()             { DU n=top; top=ss.pop(); return n; }
    __GPU__ __INLINE__ void PUSH(DU v)      { ss.push(top); top = v; }
    __GPU__ __INLINE__ void PUSH(Tensor &t) { ss.push(top); top = mmu.ten2du(t); }
    ///
    /// Forth inner interpreter
    ///
    __GPU__ int  virtual parse(char *str) override;  ///< TODO: CC - this worked, why?
    __GPU__ int  virtual number(char *str) override; ///< TODO: CC - this worked, why?

    __GPU__ int  find(const char *s);       ///< search dictionary reversely
    __GPU__ void nest();
    ///
    /// compiler proxy funtions to reduce verbosity
    ///
    __GPU__ void add_w(IU w);               ///< append a word pfa to pmem
    __GPU__ void add_iu(IU i);              ///< append an instruction unit to parameter memory
    __GPU__ void add_du(DU d);              ///< append a data unit to pmem
    __GPU__ void add_str(const char *s);    ///< append a string to pmem
    __GPU__ void call(IU w);                ///< execute word by index
};
#endif // TEN4_SRC_EFORTH_H
