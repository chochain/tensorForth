/*! @file
  @brief
  cueForth - eForth dictionary classes
*/
#ifndef CUEF_SRC_DICT_H
#define CUEF_SRC_DICT_H
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
            U16 xxx:  14;   /// reserved
            U16 len;        /// len of pfa
            IU  pidx;       /// offset to pmem space
        };
    };
    __GPU__ Code() {}      /// default constructor, called by new Vector
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
/// Forth dictionary manager
///
class Dict : public Managed {
    Code *_dict;
    U8   *_pmem;
    int  _didx = 0;
    int  _midx = 0;
    
public:
    Dict();
    ~Dict();
    ///
    /// dictionary search
    ///
    __GPU__ int  find(const char *s, bool compile, bool ucase);
    ///
    /// compiler methods
    ///
    __GPU__ Code *operator[](int i) { return (i<0) ? &_dict[_didx+i] : &_dict[i]; }
    __GPU__ int  here()             { return _midx; }
    __GPU__ void add_code(Code *c)  { _dict[_didx++] = *c; }
    __GPU__ void clear(int i)       { _didx = i; _midx = 0; }
    __GPU__ void add(U8* v, int sz) { for (; sz; sz--) _pmem[_midx++] = *v++; }
    __GPU__ void add_iu(IU i)       { add((U8*)&i, sizeof(IU)); }
    __GPU__ void add_du(DU d)       { add((U8*)&d, sizeof(DU)); }
    __GPU__ void add_str(const char *s)   { int sz = STRLENB(s)+1; sz = ALIGN2(sz); add((U8*)s, sz); }
    __GPU__ int  align()            { int i = (-_didx & 3); _didx += i; return i; }
    __GPU__ void colon(const char *name) {
        int  sz = STRLENB(name)+1;              // aligned string length
        Code *c = &_dict[_didx++];              // get next dictionary slot
    	align();                                // nfa 32-bit aligned
        c->name = (const char*)&_pmem[_midx];   // assign name field index
        c->def  = 1;                            // specify a colon word
        c->len  = 0;                            // advance counter (by number of U16)
        add((U8*)name,  ALIGN2(sz));            // setup raw name field
        c->pidx = _midx;                         // capture code field index
    }
	__GPU__ void setjmp(IU a) { wi((IU*)(pfa(-1) + a), (IU)_dict[-1].len); }
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
    /// debug methods
    ///
    __HOST__ void to_s(std::ostream &fout, IU c);
    __HOST__ void see(std::ostream &fout, IU *cp, IU *ip, int dp);
    __HOST__ void words(std::ostream &fout);
    __HOST__ void dump(std::ostream &fout, IU p0, int sz);
};
#endif // CUEF_SRC_DICT_H

