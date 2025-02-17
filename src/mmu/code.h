/** 
 * @file
 * @brief Code class - tensorForth Dictionary Entry class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_CODE_H
#define TEN4_SRC_CODE_H
///
/// CUDA functor (device only)
/// Note: nvstd::function is generic and smaller (at 56-byte)
///
///@name light-weight functor object implementation
///@brief functor object (80-byte allocated by CUDA)
///@{
struct fop { __GPU__ virtual void operator()() = 0; };  ///< functor virtual class
template<typename F>                                    ///< template functor class
struct functor : fop {
    F op;                                               ///< reference to lambda
    __GPU__ __INLINE__ functor(const F f) : op(f) {     ///< constructor
        DEBUG("F(%p) => ", this);
    }
    __GPU__ __INLINE__ void operator()() {              ///< lambda invoke
        DEBUG("F(%p).op() => ", this);
        op();
    }
};
typedef fop* FPTR;                ///< lambda function pointer
///@}
///@name Code class for dictionary word
///@{
constexpr UFP MSK_XT = (UFP)~0>>2;/// xt pointer mask (for union attributes)
struct Code : public Managed {
    const char *name = 0;         ///< name field
    union {
        FPTR xt = 0;              ///< lambda pointer (CUDA 49-bit)
        U64  *fp;                 ///< function pointer (for debugging)
        struct {
            IU  pfa;              ///< param field offset to pmem space
            U32 nfa : 16;         ///< reserved
            U32 didx: 14;         ///< dictionary index (reverse link)
            U32 imm : 1;          ///< immediate flag
            U32 udf : 1;          ///< colon defined word
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
    __GPU__ void set(const char *n, F &f, bool im) {
        name = n;
        xt   = new functor<F>(f);
        imm  = im ? 1 : 0;
        DEBUG("%cCode(name=%p, xt=%p) %s\n", im ? '*' : ' ', name, xt, n);
    }
};
#endif // TEN4_SRC_CODE_H
