/*! @file
  @brief
  cueForth - memory manager unit
*/
#ifndef CUEF_SRC_MMU_H
#define CUEF_SRC_MMU_H
#include <ostream>
#include "cuef_config.h"
#include "cuef_types.h"
#include "util.h"
///
/// CUDA functor (device only) implementation
/// Note: nvstd::function is too heavy (at 48-byte)
/// TODO: Thrust
///
struct fop {
    __GPU__  virtual void operator()(IU) = 0;
};
template<typename F>
struct functor : fop {
    union {
        F   op;             /// reference to lambda
        U64 *fp;
    };
#if CC_DEBUG
    __GPU__ functor(const F &f) : op(f) {
        printf("functor(f=%p)\n", fp);
    }
    __GPU__ void operator()(IU c) {
        printf(">> op=%p\n", fp);
        op(c);
    }
#else
    __GPU__ functor(const F &f) : op(f) {}
    __GPU__ void operator()(IU c) { op(c); }
#endif // CC_DEBUG
};
///
/// Code class for dictionary word
///
struct Code : public Managed {
    const char *name = 0;   /// name field
    union {
        fop *xt = 0;        /// lambda pointer (CUDA 49-bit)
        U64 *fp;
        struct {
            U16 def:  1;    /// colon defined word
            U16 immd: 1;    /// immediate flag
            U16 nlen: 14;   /// len of name field
            U16 plen;       /// len of pfa
            IU  pidx;       /// offset to pmem space
        };
    };
    __GPU__ Code() {}       /// default constructor, called by new Vector
    __GPU__ ~Code() {}

    template<typename F>    /// template function for lambda
#if CC_DEBUG
    __GPU__ Code(const char *n, const F &f, bool im=false) : name(n), xt(new functor<F>(f)) {
        printf("Code(...) %p: %s\n", fp, name);
        immd = im ? 1 : 0;
    }
    __GPU__ Code(const Code &c) : name(c.name), xt(c.xt) {  // called by Vector::push(T*)
        printf("Code(Code) %p: %s\n", fp, name);
    }
    __GPU__ void operator=(const Code &c) {
        printf("Code(%p:%s) << %p: %s\n", fp, name, c.fp, c.name);
        name = c.name; xt = c.xt;
    }
#else
    __GPU__ Code(const char *n, const F &f, bool im=false) : name(n), xt(new functor<F>(f)) {
        immd = im ? 1 : 0;
    }
    __GPU__ Code(const Code &c) : name(c.name), xt(c.xt) {}
    __GPU__ void operator=(const Code &c) {
        name = c.name; xt = c.xt;
    }
#endif // CC_DEBUG
};
///
/// Forth memory manager
///
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
class MMU : public Managed {
    int  _didx = 0;
    int  _midx = 0;
    Code *_dict;
    U8   *_pmem;
    DU   *_vss;

public:
    __HOST__ MMU();
    __HOST__ ~MMU();
    ///
    /// dictionary access and search methods
    ///
    __GPU__ Code &operator<<(Code *c) { _dict[_didx++] = *c; }       // initiator
    __GPU__ Code &operator[](int i)   { return (i<0) ? _dict[_didx+i] : _dict[i]; }
    __GPU__ DU*  vss(int vid) { return &_vss[vid * CUEF_SS_SZ]; }    // data stack (per VM id)
    __GPU__ U8*  mem0()       { return &_pmem[0]; }                  // base of heap space
    __GPU__ int  find(const char *s, bool compile, bool ucase);      // implemented in .cu
    ///
    /// compiler methods
    ///
    __GPU__ void colon(const char *name);                            // implemented in .cu
    __GPU__ int  align()            { int i = (-_midx & 0x3); _midx += i; return i; }
    __GPU__ int  here()             { return _midx; }
    __GPU__ void clear(int i)       { _didx = i; _midx = 0; }
    __GPU__ void add(U8* v, int sz) {
        _dict[_didx-1].plen += sz;                                   // increase parameter field length
        for (; sz; sz--) { _pmem[_midx++] = *v++; }                  // copy data to heap, TODO: dynamic parallel
    }
    __GPU__ void setjmp(IU a)       { wi((IU*)(pfa(_didx -1) + a), (IU)_dict[_didx-1].plen); }
    ///
    /// low level memory access
    ///
    __HOST__ __GPU__ U8   *pfa(IU w){ return &_pmem[_dict[w].pidx];  }
    __HOST__ __GPU__ IU   ri(IU *p) { U8 *c = (U8*)p; return ((IU)(*(c+1))<<8) | *c; }
    __HOST__ __GPU__ DU   rd(DU *p) { DU d; MEMCPY(&d, p, sizeof(DU)); return d; }
    __GPU__ DU   rd(IU w)           { return rd((DU*)&_pmem[w]); }
    __GPU__ void wd(DU *p, DU d)    { MEMCPY(p, &d, sizeof(DU)); }
    __GPU__ void wd(IU w, DU d)     { wd((DU*)&_pmem[w], d); }
    __GPU__ void wi(IU *p, IU i)    { U8 *c = (U8*)p; *c++ = i&0xff; *c = (i>>8)&0xff; }
    ///
    /// debugging methods (implemented in .cu)
    ///
    __HOST__ void to_s(std::ostream &fout, IU w);
    __HOST__ void words(std::ostream &fout);
    __HOST__ void dump(std::ostream &fout, IU p0, int sz);
    __HOST__ void see(std::ostream &fout, U8 *wp, int *i, int level);
    __HOST__ void see(std::ostream &fout, IU w);
    __HOST__ void ss_dump(std::ostream &fout, int vid, int n);
};
#endif // CUEF_SRC_MMU_H
