/** 
 * @file
 * @brief Code class - tensorForth Dictionary Entry class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __MMU_CODE_H
#define __MMU_CODE_H

namespace t4::mu {
///
///@name Code class for dictionary word
///@brief -
///  +-------------------+-------------------+
///  |    *name          |       xt          |
///  +-------------------+----+----+---------+
///                      |attr|nfa |   pfa   |
///                      +----+----+---------+
///@{
typedef   void (*FPTR)(void*);    ///< realized lambda
constexpr UFP MSK_ATTR = ~0x3;    /// xt pointer mask (for union attributes)

struct Code {
    static UFP cap;               ///< lambda captured pointer (i.e. VM*)
    const char *name = NIL;       ///< name field
    union {
        FPTR xt = NIL;            ///< lambda execution (64-bit)
        UFP  ix;                  ///< for primitives
        struct {
            U32 udf : 1;          ///< colon defined word
            U32 imm : 1;          ///< immediate flag
            U32 xx  : 6;          ///< reserved
            U32 nlen: 8;          ///< name length, NFA = pfa - nlen
            U32 didx: 16;         ///< dictionary index (reverse link)
            IU  pfa;              ///< param field offset to pmem space (32-bit)
        };
    };
    
    constexpr __HOST__ Code(const char *n, IU w) : name(n), ix((UFP)w) {} ///< primitive
    constexpr __HOST__ Code(const char *n, FPTR fp, bool im)              ///< built-in
        : name(n), xt(fp) {     
        imm = im ? 1 : 0;
        INFO("%cCode{name=%p, xt=%p} %s\n", im ? '*' : ' ', name, xt, n);
    }

    __HOST__ void set(Code &c) { name = c.name; ix = c.ix; }
    __HOST__ void exec()  {
        UFP fp = (UFP)xt & MSK_ATTR;
        if (fp && cap) {
            (*(FPTR)fp)(reinterpret_cast<void*>(cap));
        }
    }
};

} // namespace t4::mu
#endif // __MMU_CODE_H
