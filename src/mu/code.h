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
/// CUDA functor (device only)
/// Note: nvstd::function is generic and smaller (at 56-byte)
///
///@name light-weight functor object implementation
///@brief functor object (80-byte allocated by CUDA)
///@{
struct fop { __HOST__ virtual void operator()() = 0; }; ///< functor virtual class
template<typename F>                                    ///< template functor class
struct functor : fop {
    F op;                                               ///< reference to lambda
    __HOST__ __INLINE__ functor(const F f) : op(f) {    ///< constructor
        DEBUG("code#fop:%p => ", this);
    }
    __HOST__ __INLINE__ void operator()() { op(); }     ///< lambda invoke
};
typedef fop* FPTR;                ///< lambda function pointer
///@}
///@name Code class for dictionary word
///@brief -
///  +-------------------+-------------------+
///  |    *name          |       xt          |
///  +-------------------+----+----+---------+
///                      |attr|nfa |   pfa   |
///                      +----+----+---------+
///@{
constexpr UFP MSK_ATTR = ~0x3;    /// xt pointer mask (for union attributes)
struct Code : public OnHost {
    const char *name = 0;         ///< name field
    union {
        FPTR xt = 0;              ///< lambda pointer (CUDA 64-bit)
        U64  *fp;                 ///< function pointer (for debugging)
        struct {
            U32 udf : 1;          ///< colon defined word
            U32 imm : 1;          ///< immediate flag
            U32 xx  : 6;          ///< reserved
            U32 nlen: 8;          ///< name length, NFA = pfa - nlen
            U32 didx: 16;         ///< dictionary index (reverse link)
            IU  pfa;              ///< param field offset to pmem space (32-bit)
        };
    };
    __HOST__ Code(const char *n, IU w) : name(n), xt((FPTR)((UFP)w)) {}  ///< primitives
    __HOST__ ~Code() { DEBUG("Code(%s) freed\n", name); }                ///< destructor
/*
    __GPU__ Code() {}             ///< blank struct (for initilization)
    __GPU__ Code(const char *n, FPTR fp, bool im) : name(n), xt(fp) {  ///< built-in and colon words
        imm = im;
        DEBUG("%cCode(name=%p, xt=%p) %s\n", im ? '*' : ' ', name, xt, n);
    }
*/    
    template<typename F>          ///< template function for lambda
    __HOST__ void set(const char *n, F &f, bool im) {
        name = n;
        xt   = new functor<F>(f);
        imm  = im ? 1 : 0;
        DEBUG("%cCode(name=%p, xt=%p) %s\n", im ? '*' : ' ', name, xt, n);
    }
};

} // namespace t4::mu
#endif // __MMU_CODE_H
