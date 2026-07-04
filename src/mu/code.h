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
///  +-------------------+-------------------+-------------------+
///  |    *name          |       xt          |        cap        |
///  +-------------------+----+----+---------+-------------------+
///                                          |attr|nfa |   pfa   |
///                                          +----+----+---------+
///@{
typedef   void (*FPTR)(void*);    ///< realized lambda
constexpr UFP MSK_ATTR = ~0x3;    /// xt pointer mask (for union attributes)

struct Code {
    const char *name = NIL;       ///< name field
    FPTR xt = NIL;                ///< lambda execution (64-bit)
    union {
        UFP cap = 0;              ///< lambda captured pointer (VM*, 64-bit)
        struct {
            U32 udf : 1;          ///< colon defined word
            U32 imm : 1;          ///< immediate flag
            U32 xx  : 6;          ///< reserved
            U32 nlen: 8;          ///< name length, NFA = pfa - nlen
            U32 didx: 16;         ///< dictionary index (reverse link)
            IU  pfa;              ///< param field offset to pmem space (32-bit)
        };
    };
    
    __HOST__ Code(const char *n, IU w) : name(n), cap((UFP)w) {}  ///< primitive
    template<typename F> constexpr
    __HOST__ Code(const char *n, F&& f, bool im) : name(n) {      ///< built-in
        using T = typename std::decay<F>::type;                   ///< get cleaned type
        static_assert(
            sizeof(T) == sizeof(void*),
            "Error: lambda must only capture [this] (8-byte layout)"
        );
        std::swap(cap, reinterpret_cast<UFP&>(f));  /// * copy lambda capture (VM*)
        xt = [](void *data) {
            auto *fp = reinterpret_cast<T*>(&data);
            (*fp)();                                /// * execute [cap](){...}
        };
        imm = im ? 1 : 0;
        INFO("%cCode{name=%p, cap=%zx, xt=%p} %s\n", im ? '*' : ' ', name, cap, xt, n);
    }
    
    __HOST__ void set(Code &c) {
        name = c.name;
        xt   = c.xt;
        cap  = c.cap;
    }
    void exec()  {
        if (xt && (cap & MSK_ATTR)) {
            xt(reinterpret_cast<void*>(cap & MSK_ATTR));
        }
    }
};

} // namespace t4::mu
#endif // __MMU_CODE_H
