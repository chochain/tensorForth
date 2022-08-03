/** 
 * @file
 * @brief tensorForth memory manager unit
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MMU_H
#define TEN4_SRC_MMU_H
#include <curand_kernel.h>
#include "vector.h"
#include "tensor.h"
#include "tlsf.h"
///
/// CUDA functor (device only) implementation
/// Note: nvstd::function is too heavy (at sizeof(Code)=56-byte)
///
///@name light-weight functor implementation
///@brief sizeof(Code)=16
///@{
struct fop {  __GPU__  virtual void operator()() = 0; };  ///< functor virtual class
template<typename F>                         ///< template functor class
struct functor : fop {
    F op;                                    ///< reference to lambda
    __GPU__ functor(const F &f) : op(f) {    ///< constructor
        WARN("functor(%p) => ", this);
    }
    __GPU__ functor &operator=(const F &f) {
        WARN("op=%p", this);
        op = f;
        return *this;
    }
    __GPU__ void operator()() {              ///< lambda invoke
        WARN("op=%p => ", this);
        op();
    }
};
typedef fop* FPTR;          ///< lambda function pointer
/// @}
///
/// Code class for dictionary word
///
#define CODE_ATTR_FLAG      0x7
struct Code : public Managed {
    const char *name = 0;   ///< name field
    union {
        FPTR xt = 0;        ///< lambda pointer (CUDA 49-bit)
        U64  *fp;           ///< function pointer (for debugging)
        struct {
            U16 def:  1;    ///< colon defined word
            U16 immd: 1;    ///< immediate flag
            U16 diff: 1;    ///< autograd flag
            U16 xxx:  13;   ///< reserved
            IU  pfa;        ///< offset to pmem space
            U32 tidx;       ///< tensor storage offset
        };
    };
    template<typename F>    ///< template function for lambda
    __GPU__ Code(const char *n, const F &f, bool im=false) : name(n), xt(new functor<F>(f)) {
        WARN("Code(...) %p %s\n", xt, name);
        immd = im ? 1 : 0;
    }
    /*
    __GPU__ Code(const Code &c) : name(c.name), xt(c.xt) {
        WARN("Code(&c) %p %s\n", xt, name);
    }
    */
    __GPU__ Code &operator=(const Code &c) {  ///> called by Vector::push(T*)
        name = c.name;
        xt   = c.xt;
        WARN("Code()= %p %s\n", xt, name);
    }
};
///
/// macros for microcode construction
///
#define CODE(s, g)    { s, [this] __GPU__ (){ g; }}
#define IMMD(s, g)    { s, [this] __GPU__ (){ g; }, true }
typedef enum {
    UNIFORM = 0,
    NORMAL
} t4_rand_opt;
///
/// tracing level control
///
#define TRACE1(...)   { if (_trace > 0) INFO(__VA_ARGS__); }
#define TRACE2(...)   { if (_trace > 1) INFO(__VA_ARGS__); }
///
/// Forth memory manager
///
class MMU : public Managed {
    IU             _mutex = 0;      ///< lock (first so address aligned)
    IU             _didx  = 0;      ///< dictionary index
    IU             _midx  = 0;      ///< parameter memory index
    IU             _fidx  = 0;      ///< index to freed tensor list
    Code           *_dict;          ///< dictionary block
    U8             *_pmem;          ///< parameter memory block
    DU             *_vmss;          ///< VM data stack block
    U8             *_ten;           ///< tensor storage block
    DU             *_mark;          ///< list for tensors that marked free
    curandState    *_seed;          ///< for random number generator
    TLSF           _tstore;         ///< tensor storage manager
    int            _trace = 0;      ///< debug tracing verbosity level

public:
    __HOST__ MMU(int verbose=0);
    __HOST__ ~MMU();
    ///
    /// memory lock for multi-processing
    ///
    __GPU__ __INLINE__ void lock()       { MUTEX_LOCK(_mutex); }
    __GPU__ __INLINE__ void unlock()     { MUTEX_FREE(_mutex); } ///< TODO: dead lock now
    ///
    /// references to memory blocks
    ///
    __GPU__ __INLINE__ Code *dict()      { return &_dict[0]; }                      ///< dictionary pointer
    __GPU__ __INLINE__ Code *last()      { return &_dict[_didx - 1]; }              ///< last dictionary word
    __GPU__ __INLINE__ DU   *vmss(int i) { return &_vmss[i * T4_SS_SZ]; }           ///< data stack (per VM id)
    __GPU__ __INLINE__ U8   *pmem(IU i)  { return &_pmem[i]; }                      ///< base of parameter memory
    ///
    /// dictionary management ops
    ///
    __GPU__ void append(const Code *clist, int sz) {
        Code *c = (Code*)clist;
        for (int i=0; i < sz; i++) add(c++);
    }
    __GPU__ int  find(const char *s, bool compile=0, bool ucase=0);  ///< dictionary search
    __GPU__ void merge(const Code *clist, int sz);
    __GPU__ void status();
    ///
    /// compiler methods
    ///
    __GPU__ void colon(const char *name);                            ///< define colon word
    __GPU__ __INLINE__ int  align()      { int i = (-_midx & 0x3); _midx += i; return i; }
    __GPU__ __INLINE__ void clear(IU i)  { _didx = i; _midx = 0; }   ///< clear dictionary
    __GPU__ __INLINE__ void add(Code *c) { _dict[_didx++] = *c; }    ///< dictionary word assignment (deep copy)
    __GPU__ __INLINE__ void add(U8* v, int sz) {                     ///< copy data to heap, TODO: dynamic parallel
        MEMCPY(&_pmem[_midx], v, sz); _midx += sz;                   
    }
    __GPU__ __INLINE__ void setjmp(IU a) { wi(a, _midx); }           ///< set branch target address
    ///
    /// low level memory access
    ///
    __BOTH__ __INLINE__ IU   here()     { return _midx; }
    __BOTH__ __INLINE__ IU   ri(U8 *c)  { return ((IU)(*(c+1)<<8)) | *c; }
    __BOTH__ __INLINE__ IU   ri(IU pi)  { return ri(&_pmem[pi]); }
    __BOTH__ __INLINE__ DU   rd(U8 *c)  { DU d; MEMCPY(&d, c, sizeof(DU)); return d; }
    __BOTH__ __INLINE__ DU   rd(IU pi)  { return rd(&_pmem[pi]); }
    __GPU__  __INLINE__ void wd(U8 *c, DU d)   { MEMCPY(c, &d, sizeof(DU)); }
    __GPU__  __INLINE__ void wd(IU w, DU d)    { wd(&_pmem[w], d); }
    __GPU__  __INLINE__ void wi(U8 *c, IU i)   { *c++ = i&0xff; *c = (i>>8)&0xff; }
    __GPU__  __INLINE__ void wi(IU pi, IU i)   { wi(&_pmem[pi], i); }
    ///
    /// tensor life-cycle methods
    ///
#if   !T4_ENABLE_OBJ
    __GPU__  __INLINE__ void sweep()         {}             ///< holder for no object
    __GPU__  __INLINE__ void drop(DU d)      {}             ///< place holder
    __GPU__  __INLINE__ DU   dup(DU d)       { return d; }  ///< place holder
#else // T4_ENABLE_OBJ
    __GPU__  void   mark_free(DU v);                        ///< mark a tensor free
    __GPU__  void   sweep();                                ///< free marked tensor
    __GPU__  Tensor &tensor(U32 sz);                        ///< create an vector
    __GPU__  Tensor &tensor(U16 h, U16 w);                  ///< create a matrix
    __GPU__  Tensor &tensor(U16 n, U16 h, U16 w, U16 c);    ///< create a NHWC tensor
    __GPU__  void   free(Tensor &t);                        ///< free the tensor
    __GPU__  Tensor &view(Tensor &t0);                      ///< create a view to a tensor
    __GPU__  Tensor &copy(Tensor &t0);                      ///< hard copy a tensor
    __GPU__  Tensor &slice(Tensor &t0, IU x0, IU x1, IU y0, IU y1);     ///< a slice of a tensor
    __GPU__  Tensor &random(Tensor &t, t4_rand_opt ntype, int seed=0);  ///< randomize tensor cells (with given type)
    __GPU__  Tensor &scale(Tensor &t, DU v);                ///< scale a tensor
    ///
    /// short hands for eforth tensor ucodes (for DU <-> Tensor conversion)
    /// TODO: more object types
    ///
    __BOTH__ __INLINE__ Tensor &du2ten(DU d) {
        U32    *off = (U32*)&d;
        Tensor *t   = (Tensor*)(_ten + (*off & ~T4_OBJ_FLAG));
        return *t;
    }
    __BOTH__ __INLINE__ DU     ten2du(Tensor &t) {
        U32 o = ((U32)((U8*)&t - _ten)) | T4_OBJ_FLAG;
        return *(DU*)&o;
    }
    __GPU__             DU   rand(DU d, t4_rand_opt n);     ///< randomize a tensor
    __GPU__  __INLINE__ void drop(DU d) { if (IS_OBJ(d)) free(du2ten(d)); }
    __GPU__  __INLINE__ DU   dup(DU d)  { return IS_OBJ(d) ? ten2du(view(du2ten(d))) : d; }
    __GPU__  __INLINE__ DU   copy(DU d) { return IS_OBJ(d) ? ten2du(copy(du2ten(d))) : d; }
#endif // T4_ENABLE_OBJ
    ///
    /// debugging methods (implemented in .cu)
    ///
    __BOTH__ __INLINE__ int  trace()        { return _trace; }
    __BOTH__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    __HOST__ int  to_s(std::ostream &fout, IU w);
    __HOST__ void words(std::ostream &fout);
    __HOST__ void see(std::ostream &fout, U8 *p, int dp=1);     /// cannot pass pfa
    __HOST__ void see(std::ostream &fout, U16 w);
    __HOST__ void ss_dump(std::ostream &fout, U16 vid, U16 n, int radix);
    __HOST__ void mem_dump(std::ostream &fout, U16 p0, U16 sz);
};
#endif // TEN4_SRC_MMU_H
