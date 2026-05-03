/** 
 * @file
 * @brief MMU class - memory manager unit interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __MU_MMU_H
#define __MU_MMU_H
#pragma once
#include <mutex>
#include "vector.h"
#include "tensor.h"
#include "dataset.h"
#include "code.h"
#include "mpool.h"
#include "tlsf.h"

namespace t4     { class Debug; }   ///< forward declare
namespace t4::nn { class Model; }   ///< forward declare
namespace t4::mu {
///
/// Forth memory manager
/// TODO: compare TLSF to RMM (Rapids Memory Manager)
///
class MMU : public OnHost {
    IU             _didx  = 0;      ///< dictionary index
    IU             _midx  = 0;      ///< parameter memory index
    IU             _fidx  = 0;      ///< index to freed tensor list
    Code           *_dict;          ///< dictionary block
    DU             *_vmss;          ///< VM data stacks
    DU             *_vmrs;          ///< VM return stacks
    U8             *_pmem;          ///< parameter memory block
    DU             *_mark = 0;      ///< list for tensors that are marked free
    U8             *_obj  = 0;      ///< object header blocks (host memory pool)
    U8             *_data = 0;      ///< object data block (managed memory)
#if T4_DO_OBJ    
    Mpool          &_mpool;         ///< host memory pool
    TLSF           &_ostore;        ///< object storage manager
#endif // T4_DO_OBJ
    
    mutable std::mutex _mutex;      ///< lock (first so address aligned)

    __HOST__ MMU();
    __HOST__ ~MMU();
    
public:
    friend class t4::Debug;         ///< Debug can access my private members
    
    static __HOST__ MMU *get_mmu(); ///< singleton constructor/getter
    static __HOST__ void free_mmu();///< singleton destructor
    ///
    /// function pointer conversion
    ///
    __HOST__  FPTR XT(IU ioff);      ///< function pointer from _XT0 offset 
    __HOST__  IU   XTOFF(FPTR xt);   ///< _XT0 offset from function pointer
    ///
    /// references to memory blocks
    ///
    __HOST__ __INLINE__ Code *dict(IU i)  { return &_dict[i]; }          ///< dictionary pointer
    __HOST__ __INLINE__ DU   *vmss(IU i)  { return &_vmss[i*T4_SS_SZ]; } ///< dictionary pointer
    __HOST__ __INLINE__ DU   *vmrs(IU i)  { return &_vmrs[i*T4_RS_SZ]; } ///< dictionary pointer
    __HOST__ __INLINE__ U8   *pmem(IU i)  { return &_pmem[i]; }          ///< base of parameter memory
    __HOST__ __INLINE__ Code *last()      { return &_dict[_didx - 1]; }  ///< last dictionary word
    ///
    /// dictionary management ops
    ///
    __HOST__  void dict_validate();                                 ///< dictionary validation
    __HOST__  void status(bool hdr=false);                          ///< display current MMU status
    __HOST__  void dict_dump();
    __HOST__  IU   find(const char *s);                             ///< dictionary search
    ///
    /// compiler methods
    ///
    template <typename F>
    __HOST__  void add_word(const char *name, F &f, int im) {       ///< append or merge a new word
        IU   w  = find(name);                                       ///< check whether word exists
        Code &c = _dict[w ? w : _didx++];                           ///< new or exist Code object
        c.set(name, f, im);                                         /// * hardcopy Code object
        if (w) INFO("*** redefined: %s\n", c.name);
    }
    __HOST__  void colon(const char *name);                         ///< define colon word
    __HOST__  __INLINE__ int  align()      { int i = (-_midx & 0x3); _midx += i; return i; }
    __HOST__  __INLINE__ void clear(IU i)  {                        ///< clear dictionary
        _midx = _dict[i].pfa - STRLENB(_dict[i].name);
        _didx = i; 
    }
    __HOST__  __INLINE__ void add(Code *c);                         ///< create word (deep copy)
    __HOST__  __INLINE__ void add(U8* v, int sz, bool adv=true) {   ///< copy data to heap, TODO: dynamic parallel
        MEMCPY(&_pmem[_midx], v, sz); if (adv) _midx += sz;        /// * advance HERE
    }
    __HOST__  __INLINE__ void set_here(IU a) { _midx = a; }         ///< set branch target address
    ///
    /// low level memory access
    ///
    __HOST__ __INLINE__ IU   here()     { return _midx; }
    __HOST__ __INLINE__ DU   rd(U8 *c)  { return *(DU*)c; }
    __HOST__ __INLINE__ DU   rd(IU i)   {
        if (i < T4_PMEM_SZ) return rd(&_pmem[i]);
        ERROR("\nmmu.wi[%d]", i);
        return 0;
    }
    __HOST__  __INLINE__ void wd(U8 *c, DU d)   {
        DU v = rd(c);
#if T4_DO_OBJ        
        if (IS_OBJ(v)) drop(du2obj(v));                            /// * obj decref
#endif // T4_DO_OBJ        
        *(DU*)c = d;
//        MEMCPY(c, &d, sizeof(DU));
    }
    __HOST__  __INLINE__ void wd(IU i, DU d)    {
        if (i < T4_PMEM_SZ) wd(&_pmem[i], d);
        else ERROR("\nmmu.wd[%d]", i);
    }
    __HOST__  __INLINE__ void wi(U8 *c, IU n)   { *(IU*)c = n; }
    __HOST__  __INLINE__ void wi(IU i, IU n)    {
        if (i < T4_PMEM_SZ) wi(&_pmem[i], n);
        else ERROR("\nmmu.wi[%d]", i);
    }
    
#if T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ///
    /// short hands for eforth tensor ucodes (for DU <-> Tensor conversion)
    ///
//    __HOST__ U32    OBJ2X(T4Base &t) { return (U8*)&t - _obj; }         ///< object offset in _ospace
    __HOST__ U32    OBJ2X(T4Base &t) { return _mpool.offset((void*)&t); } ///< object offset in _ospace
    __HOST__ T4Base &du2obj(DU d) {                          ///< DU to Obj convertion
        U32    off = DU2X(d) & ~T4_TYPE_MSK;                 ///< clear object bit
        T4Base *t  = (T4Base*)(_obj + off);                  ///< convert to object pointer
        return *t;
    }
    __HOST__ DU     obj2du(T4Base &t) {                      ///< conver Obj to DU
        U32 o = OBJ2X(t) | T4_TT_OBJ;                        ///< mark object bit
        return *(DU*)&o;                                     ///< convert to DU value
    }
    ///
    /// tensor object life-cycle methods
    ///
    __HOST__  void   sweep();                                ///< free marked tensor
    __HOST__  void   drop(T4Base &t0);                       ///< reduce ref counter
    __HOST__  void   mark_free(DU v);                        ///< mark an object to be freed in host
    __HOST__  Tensor &talloc(U64 sz);                        ///< allocate from tensor space
    __HOST__  Tensor &tensor(U64 sz);                        ///< create an vector
    __HOST__  Tensor &tensor(U32 h, U32 w);                  ///< create a matrix
    __HOST__  Tensor &tensor(U32 n, U32 h, U32 w, U32 c);    ///< create a NHWC tensor
    __HOST__  void   resize(Tensor &t, U64 sz);              ///< resize the tensor storage
    __HOST__  void   free(Tensor &t);                        ///< free the tensor
    __HOST__  Tensor &copy(Tensor &t0);                      ///< hard copy a tensor
    __HOST__  Tensor &dim(Tensor &t0);                       ///< dimensions
    __HOST__  Tensor &slice(Tensor &t0, IU x0, IU x1, IU y0, IU y1);     ///< a slice of a tensor
#if T4_DO_NN
    ///
    /// neaural network objects
    ///
    __HOST__  Dataset    &dataset(U32 batch_sz);                 ///< create a NN dataset
    __HOST__  nn::Model  &model(int &trace, U32 nsz=T4_NET_SZ);  ///< create a NN model (nsz=max layers)
    __HOST__  void       free(nn::Model &m);
#endif // T4_DO_NN
#else  // !T4_DO_OBJ ==========================================================
    __HOST__  void   sweep()    {}                           ///< holder for no object
    
#endif // T4_DO_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
};

} // namespace t4::mu
#endif // __MU_MMU_H

