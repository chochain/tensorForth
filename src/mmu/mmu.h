/** 
 * @file
 * @brief MMU class - memory manager unit interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_MMU_H
#define TEN4_SRC_MMU_H
#include <curand_kernel.h>
#include "vector.h"

//#include "tensor.h"
//#include "tlsf.h"
///
/// CUDA functor (device only)
/// Note: nvstd::function is generic and smaller (at 56-byte)
///
///@name light-weight functor object implementation
///@brief functor object (80-byte allocated by CUDA)
///@{
struct fop { __GPU__ virtual void operator()() = 0; };  ///< functor virtual class
template<typename F>                         ///< template functor class
struct functor : fop {
    F op;                                    ///< reference to lambda
    __GPU__ functor(const F f) : op(f) {     ///< constructor
        WARN("F(%p) => ", this);
    }
    __GPU__ __INLINE__ void operator()() {   ///< lambda invoke
        WARN("F(%p).op() => ", this);
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
    template<typename F>    ///< template function for lambda
    __GPU__ Code(const char *n, F f, bool im) : name(n), xt(new functor<F>(f)) {
        immd = im ? 1 : 0;
        WARN("%cCode(name=%p, xt=%p) %s\n", im ? '*' : ' ', name, xt, n);
    }
    /* Note: no update (construct only)
    __GPU__ Code(const Code &c) : name(c.name), xt(c.xt), u(c.u) {
        WARN("Code(&c) %p %s\n", xt, name);
    }
    __GPU__ Code &operator=(const Code &c) {  ///> called by Vector::push(T*)
        name = c.name;
        xt   = c.xt;
        u    = c.u;
        WARN("Code= %p %s\n", xt, name);
        return *this;
    }
    */
};
///
/// macros for microcode construction
///
#define ADD_CODE(s, g, im) {                     \
    Code c = { s, [this] __GPU__ (){ g; }, im }; \
    mmu.add(&c);                                 \
}
#define CODE(s, g) ADD_CODE(s, g, false)
#define IMMD(s, g) ADD_CODE(s, g, true)

typedef enum {
    UNIFORM = 0,
    NORMAL
} t4_rand_opt;
///
/// tracing level control
///
#define MM_TRACE1(...) { if (_trace > 0) INFO(__VA_ARGS__); }
#define MM_TRACE2(...) { if (_trace > 1) INFO(__VA_ARGS__); }
///
/// Forth memory manager
/// TODO: compare TLSF to RMM (Rapids Memory Manager)
///
struct Model;
struct Dataset;
class MMU : public Managed {
    int            _khz;            ///< GPU clock speed
    int            _trace = 0;      ///< debug tracing verbosity level
    IU             _mutex = 0;      ///< lock (first so address aligned)
    IU             _didx  = 0;      ///< dictionary index
    IU             _midx  = 0;      ///< parameter memory index
    IU             _fidx  = 0;      ///< index to freed tensor list
    Code           *_dict;          ///< dictionary block
    U8             *_pmem;          ///< parameter memory block
    DU             *_vmss;          ///< VM data stack block
    curandState    *_seed;          ///< for random number generator
    DU             *_mark = 0;      ///< list for tensors that marked free
    U8             *_obj  = 0;      ///< object storage block
#if T4_ENABLE_OBJ    
    TLSF           _ostore;         ///< object storage manager
#endif // T4_ENABLE_OBJ    

public:
    __HOST__ MMU(int khz, int verbose=0);
    __HOST__ ~MMU();
    ///
    /// memory lock for multi-processing
    ///
    __GPU__ __INLINE__ void lock()       { MUTEX_LOCK(_mutex); }
    __GPU__ __INLINE__ void unlock()     { MUTEX_FREE(_mutex); } ///< TODO: dead lock now
    ///
    /// references to memory blocks
    ///
    __GPU__ __INLINE__ Code *dict()      { return &_dict[0]; }            ///< dictionary pointer
    __GPU__ __INLINE__ Code *last()      { return &_dict[_didx - 1]; }    ///< last dictionary word
    __GPU__ __INLINE__ DU   *vmss(int i) { return &_vmss[i * T4_SS_SZ]; } ///< data stack (per VM id)
    __GPU__ __INLINE__ U8   *pmem(IU i)  { return &_pmem[i]; }            ///< base of parameter memory
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
    __GPU__ __INLINE__ void clear(IU i)  {                           ///< clear dictionary
        _didx = i; _midx = _dict[i].nfa;
    }
    __GPU__ __INLINE__ void add(Code *c) { _dict[_didx++] = *c; }    ///< dictionary word assignment (deep copy)
    __GPU__ __INLINE__ void add(U8* v, int sz, bool adv=true) {      ///< copy data to heap, TODO: dynamic parallel
        MEMCPY(&_pmem[_midx], v, sz); if (adv) _midx += sz;          /// * advance HERE
    }
    __GPU__ __INLINE__ void setjmp(IU a) { wi(a, _midx); }           ///< set branch target address
    ///
    /// low level memory access
    ///
    __BOTH__ __INLINE__ IU   here()     { return _midx; }
    __BOTH__ __INLINE__ IU   ri(U8 *c)  { return ((IU)(*(c+1)<<8)) | *c; }
    __BOTH__ __INLINE__ IU   ri(IU i)  {
        if (i < T4_PMEM_SZ) return ri(&_pmem[i]);
        ERROR("\nmmu.wi[%d]", i);
        return 0;
    }
    __BOTH__ __INLINE__ DU   rd(U8 *c)  { DU d; MEMCPY(&d, c, sizeof(DU)); return d; }
    __BOTH__ __INLINE__ DU   rd(IU i)  {
        if (i < T4_PMEM_SZ) return rd(&_pmem[i]);
        ERROR("\nmmu.wi[%d]", i);
        return 0;
    }
    __GPU__  __INLINE__ void wd(U8 *c, DU d)   { MEMCPY(c, &d, sizeof(DU)); }
    __GPU__  __INLINE__ void wd(IU i, DU d)    {
        if (i < T4_PMEM_SZ) wd(&_pmem[i], d);
        else ERROR("\nmmu.wd[%d]", i);
    }
    __GPU__  __INLINE__ void wi(U8 *c, IU n)   { *c++ = n&0xff; *c = (n>>8)&0xff; }
    __GPU__  __INLINE__ void wi(IU i, IU n)   {
        if (i < T4_PMEM_SZ) wi(&_pmem[i], n);
        else ERROR("\nmmu.wi[%d]", i);
    }
    ///
    /// tensor life-cycle methods
    ///
#if T4_ENABLE_OBJ
    __HOST__ int  to_s(std::ostream &fout, DU s);                ///< dump object from descriptor
    __HOST__ int  to_s(std::ostream &fout, Tensor &t);           ///< dump object on stack
    __BOTH__ T4Base &du2obj(DU d) {
        U32    *off = (U32*)&d;
        T4Base *t   = (T4Base*)(_obj + (*off & ~T4_OBJ_FLAG));
        return *t;
    }
    __BOTH__ DU obj2du(T4Base &t) {
        U32 o = ((U32)((U8*)&t - _obj)) | T4_OBJ_FLAG;
        return *(DU*)&o;
    }
    __BOTH__ int ref_inc(DU d) { return du2obj(d).ref_inc(); }
    __BOTH__ int ref_dec(DU d) { return du2obj(d).ref_dec(); }
    
    __GPU__  void   mark_free(T4Base &t);                   ///< mark a tensor free
    __GPU__  void   mark_free(DU v);                        ///< mark a tensor free
    __GPU__  void   sweep();                                ///< free marked tensor
    __GPU__  Tensor &talloc(U32 sz);                        ///< allocate from tensor space
    __GPU__  Tensor &tensor(U32 sz);                        ///< create an vector
    __GPU__  Tensor &tensor(U16 h, U16 w);                  ///< create a matrix
    __GPU__  Tensor &tensor(U16 n, U16 h, U16 w, U16 c);    ///< create a NHWC tensor
    __GPU__  Model  &model(U32 sz=T4_NET_SZ);               ///< create a NN model
    __GPU__  Dataset&dataset(U16 batch_sz);                 ///< create a NN dataset
    __GPU__  void   resize(Tensor &t, U32 sz);              ///< resize the tensor storage
    __GPU__  void   free(Tensor &t);                        ///< free the tensor
    __GPU__  void   free(Model &m);
    __GPU__  Tensor &view(Tensor &t0);                      ///< create a view to a tensor
    __GPU__  Tensor &copy(Tensor &t0);                      ///< hard copy a tensor
    __GPU__  Tensor &slice(Tensor &t0, IU x0, IU x1, IU y0, IU y1);     ///< a slice of a tensor
    __GPU__  Tensor &random(Tensor &t, t4_rand_opt ntype, DU bias=DU0, DU scale=DU1);  ///< randomize tensor cells (with given type)
    ///
    /// short hands for eforth tensor ucodes (for DU <-> Object conversion)
    ///
    __GPU__  DU     dup(DU d);
    __GPU__  DU     view(DU d);
    __GPU__  DU     copy(DU d);
    __GPU__  void   drop(DU d);
    __GPU__  DU     rand(DU d, t4_rand_opt n);             ///< randomize a tensor
#else  // T4_ENABLE_OBJ
    __GPU__  __INLINE__ void sweep()    {}                  ///< holder for no object
    __GPU__  __INLINE__ void drop(DU d) {}                  ///< place holder
    __GPU__  __INLINE__ DU   dup(DU d)  { return d; }       ///< place holder
#endif // T4_ENABLE_OBJ
    ///
    /// debugging methods (implemented in .cu)
    ///
    __GPU__  __INLINE__ DU   ms()           { return static_cast<double>(clock64()) / _khz; }
    __BOTH__ __INLINE__ int  khz()          { return _khz;   }
    __BOTH__ __INLINE__ int  trace()        { return _trace; }
    __BOTH__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    
    __HOST__ int  to_s(std::ostream &fout, IU w);                ///< dump word 
    __HOST__ void words(std::ostream &fout);                     ///< display dictionary
    __HOST__ void see(std::ostream &fout, U8 *p, int dp=1);      ///< disassemble a word
    __HOST__ void see(std::ostream &fout, U16 w);               
    __HOST__ void ss_dump(std::ostream &fout, U16 vid, U16 n, int radix);
    __HOST__ void mem_dump(std::ostream &fout, U16 p0, U16 sz);   ///< dump a section of param memory
};
#endif // TEN4_SRC_MMU_H
