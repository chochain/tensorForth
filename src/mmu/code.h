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
/// @}
typedef fop* FPTR;          ///< lambda function pointer
///
/// Code class for dictionary word
///
#define CODE_ATTR_FLAG      0x3              /**< nVidia func 4-byte aligned */
struct Code : public Managed {
    const char *name = 0;   ///< name field
    union {
        FPTR xt = 0;        ///< lambda pointer (CUDA 49-bit)
        U64  *fp;           ///< function pointer (for debugging)
        struct {
            U16 colon: 1;   ///< colon defined word
            U16 immd:  1;   ///< immediate flag
            U16 xx1:   1;   ///< reserved 1
            U16 xx2:   1;   ///< reserved 2
            U16 didx:  12;  ///< dictionary index (4K links)
            IU  pfa;        ///< parameter field offset in pmem space
            IU  nfa;        ///< name field offset to pmem space
            U16 xx3;        ///< reserved
        };
    };
    
    /* Note: no constructor needed
    template<typename F>    ///< template function for lambda
    __GPU__ Code(const char *n, F f, bool im) : name(n), xt(new functor<F>(f)) {
        immd = im ? 1 : 0;
        DEBUG("%cCode(name=%p, xt=%p) %s\n", im ? '*' : ' ', name, xt, n);
    }
    __GPU__ Code(const Code &c) : name(c.name), xt(c.xt), u(c.u) {
        DEBUG("Code(&c) %p %s\n", xt, name);
    }
    __GPU__ Code &operator=(const Code &c) {  ///> called by Vector::push(T*)
        name = c.name;
        xt   = c.xt;
        u    = c.u;
        DEBUG("Code= %p %s\n", xt, name);
        return *this;
    }
    */
    ~Code() { DEBUG("Code(%s) freed\n", name); }
    
    template<typename F>    ///< template function for lambda
    __GPU__ void set(const char *n, F &f, bool im) {
        name = n;
        xt   = new functor<F>(f);
        immd = im ? 1 : 0;
        DEBUG("%cCode(name=%p, xt=%p) %s\n", im ? '*' : ' ', name, xt, n);
    }
};
#endif // TEN4_SRC_CODE_H
