/** 
 * @file
 * @brief tensorForth memory manager unit
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MMU_H
#define TEN4_SRC_MMU_H
#include <ostream>
#include "ten4_config.h"
#include "ten4_types.h"
#include "util.h"
///
/// CUDA functor (device only) implementation
/// Note: nvstd::function is too heavy (at 48-byte)
///
///@name functor implementation
///@{
struct fop {  __GPU__  virtual void operator()() = 0; };  ///< functor virtual class
template<typename F>        ///< template functor class
struct functor : fop {
    union {
        F   op;             ///< reference to lambda
        U64 *fp;            ///< function pointer for debugging print
    };
    __GPU__ functor(const F &f) : op(f) {
        MMU_DEBUG("functor(f=%p)\n", fp);
    }
    __GPU__ __INLINE__ void operator()() {
        MMU_DEBUG(">> op=%p\n", fp);
        op();
    }
};
typedef fop* FPTR;          ///< lambda function pointer
/// @}
///
/// Code class for dictionary word
///
struct Code : public Managed {
    const char *name = 0;   ///< name field
    union {
        FPTR xt = 0;        ///< lambda pointer (CUDA 49-bit)
        U64 *fp;
        struct {
            U16 def:  1;    ///< colon defined word
            U16 immd: 1;    ///< immediate flag
            U16 xxx: 14;    ///< reserved
            IU  pfa;        ///< offset to pmem space
        };
    };
    template<typename F>    ///< template function for lambda
    __GPU__ Code(const char *n, const F &f, bool im=false) : name(n), xt(new functor<F>(f)) {
        MMU_DEBUG("Code(...) %p: %s\n", fp, name);
        immd = im ? 1 : 0;
    }
    __GPU__ Code(const Code &c) : name(c.name), xt(c.xt) {  ///> called by Vector::push(T*)
        MMU_DEBUG("Code(Code) %p: %s\n", fp, name);
    }
};
#define CODE(s, g)    { s, [this] __GPU__ (){ g; }}
#define IMMD(s, g)    { s, [this] __GPU__ (){ g; }, true }
///
/// Forth memory manager
///
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
class MMU : public Managed {
    IU   _didx = 0;
    IU   _midx = 0;
    Code *_dict;
    U8   *_pmem;
    DU   *_vss;

public:
    __HOST__ MMU();
    __HOST__ ~MMU();
    ///
    /// dictionary access and search methods
    ///
    __GPU__ __INLINE__ Code &operator<<(Code *c) { return _dict[_didx++] = *c; }    ///< dictionary word assignment
    __GPU__ __INLINE__ Code &operator[](int i)   { return (i<0) ? _dict[_didx+i] : _dict[i]; } ///< dictionary accessor by index

    __GPU__ __INLINE__ Code *dict()      { return &_dict[0]; }                      ///< dictionary pointer
    __GPU__ __INLINE__ Code *last()      { return &_dict[_didx - 1]; }              ///< last dictionary word
    __GPU__ __INLINE__ DU*  vss(int vid) { return &_vss[vid * T4_SS_SZ]; }          ///< data stack (per VM id)
    __GPU__ __INLINE__ U8*  mem(IU pi)   { return &_pmem[pi]; }                     ///< base of heap space

    __GPU__ int  find(const char *s, bool compile, bool ucase);      ///> implemented in .cu
    ///
    /// compiler methods
    ///
    __GPU__ void colon(const char *name);                            ///> implemented in .cu
    __GPU__ __INLINE__ int  align()      { int i = (-_midx & 0x3); _midx += i; return i; }
    __GPU__ __INLINE__ void clear(IU i)  { _didx = i; _midx = 0; }
    __GPU__ __INLINE__ void add(U8* v, int sz) {
        MEMCPY(&_pmem[_midx], v, sz); _midx += sz;                   ///> copy data to heap, TODO: dynamic parallel
    }
    __GPU__ __INLINE__ void setjmp(IU a) { wi(a, _midx); }           ///> set branch target address
    ///
    /// low level memory access
    ///
    __HOST__ __GPU__ __INLINE__ IU here()     { return _midx; }
    __HOST__ __GPU__ __INLINE__ IU ri(U8 *c)  { return ((IU)(*(c+1)<<8)) | *c; }
    __HOST__ __GPU__ __INLINE__ IU ri(IU pi)  { return ri(&_pmem[pi]); }
    __HOST__ __GPU__ __INLINE__ DU rd(U8 *c)  { DU d; MEMCPY(&d, c, sizeof(DU)); return d; }
    __HOST__ __GPU__ __INLINE__ DU rd(IU pi)  { return rd(&_pmem[pi]); }
    __GPU__ __INLINE__ void wd(U8 *c, DU d)   { MEMCPY(c, &d, sizeof(DU)); }
    __GPU__ __INLINE__ void wd(IU w, DU d)    { wd(&_pmem[w], d); }
    __GPU__ __INLINE__ void wi(U8 *c, IU i)   { *c++ = i&0xff; *c = (i>>8)&0xff; }
    __GPU__ __INLINE__ void wi(IU pi, IU i)   { wi(&_pmem[pi], i); }
    ///
    /// debugging methods (implemented in .cu)
    ///
    __HOST__ void to_s(std::ostream &fout, IU w);
    __HOST__ void words(std::ostream &fout);
    __HOST__ void see(std::ostream &fout, U8 *p, int dp=1);     /// cannot pass pfa
    __HOST__ void see(std::ostream &fout, U16 w);
    __HOST__ void ss_dump(std::ostream &fout, U16 vid, U16 n, int radix);
    __HOST__ void mem_dump(std::ostream &fout, U16 p0, U16 sz);
};
#endif // TEN4_SRC_MMU_H
