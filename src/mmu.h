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
struct fop {  __GPU__  virtual void operator()(IU) = 0; };
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
    __GPU__ __INLINE__ void operator()(IU c) { op(c); }
#endif // CC_DEBUG
};
typedef fop* FPTR;          /// lambda function pointer
///
/// Code class for dictionary word
///
struct Code : public Managed {
    const char *name = 0;   /// name field
    union {
        FPTR xt = 0;        /// lambda pointer (CUDA 49-bit)
        U64 *fp;
        struct {
            U16 def:  1;    /// colon defined word
            U16 immd: 1;    /// immediate flag
            U16 xxx: 14;    /// reserved
            IU  pfa;        /// offset to pmem space
        };
    };
    __GPU__ Code()  {}      /// default constructor, called by new Vector
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
#define CODE(s, g)    { s, [this] __GPU__ (IU c){ g; }}
#define IMMD(s, g)    { s, [this] __GPU__ (IU c){ g; }, true }
///
/// Forth memory manager
///
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
class MMU : public Managed {
    IU   _didx = 0;
    IU   _midx = 0;
    UFP  _xt0  = ~0;
    Code *_dict;
    U8   *_pmem;
    DU   *_vss;

public:
    __HOST__ MMU();
    __HOST__ ~MMU();
    ///
    /// dictionary access and search methods
    ///
    __GPU__ __INLINE__ Code &operator<<(Code *c) { _dict[_didx++] = *c; if ((UFP)c->xt < _xt0) _xt0 = (UFP)c->xt; }  /// assignment
    __GPU__ __INLINE__ Code &operator[](int i)   { return (i<0) ? _dict[_didx+i] : _dict[i]; }                       /// fetch

    __GPU__ __INLINE__ Code *dict()      { return &_dict[0]; }                      /// dictionary pointer
    __GPU__ __INLINE__ Code *last()      { return &_dict[_didx - 1]; }              /// last dictionary word
    __GPU__ __INLINE__ DU*  vss(int vid) { return &_vss[vid * CU4_SS_SZ]; }         /// data stack (per VM id)
    __GPU__ __INLINE__ IU   here()       { return _midx; }
    __GPU__ __INLINE__ U8*  mem(IU pi)   { return &_pmem[pi]; }                     /// base of heap space
    __GPU__ __INLINE__ IU   xtoff(UFP ix){ return (IU)(ix - _xt0); }                /// offset to code space
    __GPU__ __INLINE__ UFP  xt(IU ix)    { return _xt0 + (ix & ~0x3); }             /// convert index to function pointer

    __GPU__ int  find(const char *s, bool compile, bool ucase);      ///> implemented in .cu
    ///
    /// compiler methods
    ///
    __GPU__ void colon(const char *name);                            ///> implemented in .cu
    __GPU__ __INLINE__ int  align()     { int i = (-_midx & 0x3); _midx += i; return i; }
    __GPU__ __INLINE__ void clear(IU i) { _didx = i; _midx = 0; }
    __GPU__ __INLINE__ void add(U8* v, int sz) {
        MEMCPY(&_pmem[_midx], v, sz); _midx += sz;                   ///> copy data to heap, TODO: dynamic parallel
    }
    __GPU__ __INLINE__ void setjmp(IU a) { wi(a, _midx); }           ///> set branch target address
    ///
    /// low level memory access
    ///
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
    __HOST__ int  pfa2word(IU pi);
    __HOST__ void to_s(std::ostream &fout, IU w);
    __HOST__ void words(std::ostream &fout);
    __HOST__ void see(std::ostream &fout, U8 *p, int dp=1);
    __HOST__ void see(std::ostream &fout, U16 w);
    __HOST__ void ss_dump(std::ostream &fout, U16 vid, U16 n);
    __HOST__ void mem_dump(std::ostream &fout, U16 p0, U16 sz);
};
#endif // CUEF_SRC_MMU_H
