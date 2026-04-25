/** -*- c++ -*-
 * @file
 * @brief MMU class - memory manager implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>             /// setw, setbase
#include "mmu.h"
#include "dataset.h"
#include "nn/model.h"

namespace t4::mu {
using nn::Model;
///
/// memory lock for multi-processing
///
#define LOCK()   std::unique_lock<std::mutex> lock(_mutex)
#define UNLOCK() lock.unlock()
///
///@name static class member
///@note: CUDA does not support device static data
///@{
MMU *_mmu = NULL;              ///< singleton MMU controler
UFP _XT0;
UFP _NM0;
///@}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
#if !T4_DO_OBJ
MMU::MMU() {
    
#else // T4_DO_OBJ
MMU::MMU() : _mpool(Mpool::get_instance()), _ostore(TLSF::get_instance()) {
#if T4_DO_NN
    _obj = (U8*)_mpool.init(sizeof(Dataset), T4_MPOOL_SZ);
#else  // !T4_DO_NN
    _obj = (U8*)_mpool.init(sizeof(Tensor), T4_MPOOL_SZ);
#endif // T4_DO_NN
    
    H_ALLOC(&_mark,  sizeof(DU) * T4_TFREE_SZ);
    MM_ALLOC(&_data,  T4_OSTORE_SZ);              /// * object store in Managed Memory
    _ostore.init(_data, T4_OSTORE_SZ);
#endif // T4_DO_OBJ
    
    H_ALLOC(&_dict, sizeof(Code) * T4_DICT_SZ);
    H_ALLOC(&_vmss, sizeof(DU) * T4_SS_SZ * T4_VM_COUNT);
    H_ALLOC(&_vmrs, sizeof(DU) * T4_RS_SZ * T4_VM_COUNT);
    H_ALLOC(&_pmem, T4_PMEM_SZ);

    _midx = T4_USER_AREA;      /// set aside user area (for base and maybe compile)
    
    TRACE(
        "\\ MMU:\n"
        "\\\tdict=%p\n"
        "\\\tvmss=%p\n"
        "\\\tvmrs=%p\n"
        "\\\tmem =%p\n"
        "\\\tmark=%p\n"
        "\\\tobj =%p\n"
        "\\\tdata=%p (Managed)\n",
        _dict, _vmss, _vmrs, _pmem, _mark, _obj, _data);
}
__HOST__
MMU::~MMU() {
    if (_data) MM_FREE(_data);
    if (_mark) H_FREE((void*)_mark);
    H_FREE(_pmem);
    H_FREE(_vmrs);
    H_FREE(_vmss);
    H_FREE(_dict);
    TRACE("\\   MMU: memory freed\n");
}
__HOST__ MMU*
MMU::get_mmu() {
    if (!_mmu) _mmu = new MMU();
    return _mmu;
}
__HOST__ void
MMU::free_mmu() {
    if (_mmu) delete _mmu;
}
///
/// static functions (for type conversion)
///
__HOST__  FPTR MMU::XT(IU ioff)      { return (FPTR)(_XT0 + ioff);  }
__HOST__  IU   MMU::XTOFF(FPTR xt)   { return (IU)((UFP)xt - _XT0); }
///
/// dictionary management methods
/// TODO: use const Code[] directly, as ROM, to prevent deep copy
///
__HOST__ void
MMU::dict_validate() {
    UFP  x0 = ~0;                           ///< base of xt   allocations
    UFP  n0 = ~0;
    Code *c = _dict;
    for (int i=0; i < _didx; i++, c++) {    /// * scan thru for max range
        if ((UFP)c->xt   < x0) x0 = (UFP)c->xt;
        if ((UFP)c->name < n0) n0 = (UFP)c->name;
    }
    _XT0 = x0;
    _NM0 = n0;
    _dict[0].xt = (FPTR)x0;                 /// * borrow for xt0
}

__HOST__ IU
MMU::find(const char *s) {
    IU v = 0;
    DEBUG("mmu.find(%s) => ", s);
    for (IU i = _didx - 1; _didx && !v && i > 0; --i) {
        if (STRCMP(_dict[i].name, s)==0) v = i;
    }
    return v;
}

__HOST__  __INLINE__ void
MMU::add(Code *c) {                         ///< create word (deep copy)
    LOCK();
    _dict[_didx++] = *c;
}

__HOST__ void
MMU::status(bool hdr) {
    if (hdr) {
        INFO("\\ MMU.stat dict[%d/%d], pmem[%d]=%0.1f%%, tfree[%d/%d]\n",
             _didx, T4_DICT_SZ, _midx, 100.0*(_midx/T4_PMEM_SZ), _fidx, T4_TFREE_SZ);
    }
    ///
    /// display object store statistics
    ///
#if T4_DO_OBJ
    _mpool.status();
    _ostore.status();
#endif // T4_DO_OBJ
}

__HOST__ void
MMU::dict_dump() {
    Code *c = _dict;
    INFO("Built-in Dictionary [name0=0x%lx, xt0=0x%lx]\n", _NM0, _XT0);
    for (int i=0; i<_didx; i++, c++) {      ///< dump dictionary from device
        IU  ix  = c->udf ? c->pfa : (UFP)c->xt - _XT0;
        INFO("%4d|%03x> name=%6x, %s=%6x %s\n", i, i,
             c->udf ? (c->pfa - c->nlen) : (U32)((UFP)c->name - _NM0),
             c->udf ? "pf" : "xt", ix, c->name);
    }
}
///
/// colon - dictionary word compiler
///
__HOST__ void
MMU::colon(const char *name) {
    MM_DB("colon(%s) => ", name);
    int  nsz = ALIGN(STRLENB(name) + 1);    /// aligned string length
    Code &c = _dict[_didx++];               /// get next dictionary slot
    align();                                /// nfa 32-bit aligned (adjust _midx)
    c.udf  = 1;                             /// specify a colon word
    c.nlen = nsz;                           /// name field offset (include '\0')
    c.didx = _didx-1;                       /// directory index (reverse link)
    c.name = (const char*)&_pmem[_midx];    /// assign name field index
    add((U8*)name,  nsz);                   /// setup raw name field
    c.pfa  = _midx;                         /// parameter field offset
}
///
/// tensor life-cycle methods
///
#if T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
__HOST__ void                      ///< release marked free tensor
MMU::sweep() {
    LOCK();
    for (int i = 0, n=_fidx; n && i < n; i++) {
        DU v = _mark[i];
        MM_DB("mmu#release T:%x from free[%d]\n", DU2X(v) & ~T4_TT_OBJ, i);
        drop(du2obj(v));
    }
    _fidx = 0;
}
__HOST__ void
MMU::drop(T4Base &t) {
#if T4_DO_NN
    if (t.is_model())  { free((Model&)t); return;  }   /// release TLSF memory block
#endif  // T4_DO_NN
    free((Tensor&)t);             /// check reference count
}

__HOST__ void
MMU::mark_free(DU v) {            ///< mark a tensor free for release
    if (IS_VIEW(v)) return;
    T4Base &t = du2obj(v);
    MM_DB("mmu#mark T:%x to free[%d]\n", OBJ2X(t), _fidx);

    LOCK();
    if (_fidx < T4_TFREE_SZ) _mark[_fidx++] = obj2du(t);
    else ERROR("ERR: tfree store full, increase T4_TFREE_SZ!");
}

__HOST__ Tensor&                    ///< allocate a tensor from tensor space
MMU::talloc(U64 sz) {
    MM_DB("mmu#talloc(0x%lx) {\n", sz);
    Tensor *t = (Tensor*)_mpool.malloc();   /// * was = *(Tensor*)_ostore.malloc(sizeof(Tensor));
    void   *d = _ostore.malloc(sz * sizeof(DU));
    MM_DB("} mmu#talloc => T:%x+%x\n", OBJ2X(*t), (U32)((U8*)d - _data));
    t->reset(d, sz);
    status();
    return *t;
}
__HOST__ Tensor&                    ///< create a one-dimensional tensor
MMU::tensor(U64 sz) {
    MM_DB("mmu#tensor(%ld) numel=%ld ", sz, sz);
    return talloc(sz);
}
__HOST__ Tensor&                    ///< create a 2-dimensional tensor
MMU::tensor(U32 h, U32 w) {
    U64 sz = (U64)h * w;
    MM_DB("mmu#tensor(%d,%d) numel=%ld ", h, w, sz);
    Tensor &t = talloc(sz);
    t.reshape(h, w);
    return t;
}
__HOST__ Tensor&                    ///< create a NHWC tensor
MMU::tensor(U32 n, U32 h, U32 w, U32 c) {
    U64 sz = (U64)n * h * w * c;
    MM_DB("mmu#tensor(%d,%d,%d,%d) numel=%ld ", n, h, w, c, sz);
    Tensor &t = talloc(sz);
    t.reshape(n, h, w, c);
    return t;
}
__HOST__ void
MMU::resize(Tensor &t, U64 sz) {
    if (t.rank != 1) { ERROR("mmu#resize rank==1 only\n"); return; }
    MM_DB("mmu#resize numel=%ld (was %ld) ", sz, t.numel);
    DU *d0 = t.data;             /// * keep original memory block
    t.data = (DU*)_ostore.malloc(sz * sizeof(DU));
    ///
    /// hardcopy tensor storage
    ///
    memcpy(t.data, d0, (t.numel < sz ? t.numel : sz) * sizeof(DU));
    t.H() = t.numel = sz;        /// * adjust tensor storage size
    
    _ostore.free(d0);            /// * release 
    status();
}
__HOST__ void                    ///< release tensor memory blocks
MMU::free(Tensor &t) {
    int n = t.rank;
    MM_DB("mmu#free(T%d) numel=%ld T:%x {\n", n, t.numel, OBJ2X(t));
    _ostore.free(t.data);        /// * free physical data
    if (t.grad_fn != L_NONE) {
        MM_DB("{\n");
        for (int i=0; t.mtum[i] && i < 4; i++) {
            if (t.mtum[i] == t.grad[i]) continue;   /// * dummy pointers for SGD
            MM_DB("\t\t"); free(*t.mtum[i]);
        }
        for (int i=0; t.grad[i] && i < 4; i++) {
            MM_DB("\t\t"); free(*t.grad[i]);    /// recursive
        }
        MM_DB("\t} ");
    }
    _mpool.free(&t);              /// * free tensor object itself
    status();
    MM_DB("} mmu#free(T%d)\n", n);
}
///
/// deep copy a tensor
/// TODO: CDP
///
__HOST__ Tensor&
MMU::copy(Tensor &t0) {
    if (!t0.is_tensor()) return t0;         ///> skip, TODO: copy model

    MM_DB("mmu#copy(T%d:%x) numel=%ld {\n", t0.rank, OBJ2X(t0), t0.numel);
    Tensor *t1 = (Tensor*)_mpool.malloc();  /// * was = *(Tensor*)_ostore.malloc(sizeof(Tensor));
    memcpy(t1, &t0, sizeof(Tensor));        /// * copy attributes
    ///
    /// set attributes
    ///
    for (int i=0; i<5; i++) t1->grad[i] = t1->mtum[i] = NULL;  /// * blank gradients
    t1->grad_fn = L_NONE;                   /// * not a network layer
    t1->nref    = 1;                        /// * reset ref counter
    ///
    /// hard copy data block
    ///
    U64 bsz = sizeof(DU) * t0.numel;
    t1->data = (DU*)_ostore.malloc(bsz);
    *t1 = t0;                               /// * copy all tensor elements
    
    MM_DB("} mmu#copy(T%d) => T%d:%x\n", t0.rank, t1->rank, OBJ2X(*t1));
    return *t1;
}
__HOST__ Tensor&
MMU::dim(Tensor &t0) {
    const int map[] = { 3, 0, 1, 2 };       /// HWCN => NHWC
    Tensor &t = tensor(4);
    for (int i=0; i<4; i++) t[i] = t0[map[i]];
    return t;
}
///
/// tensor slice & dice
/// TODO: CDP
///
__HOST__ Tensor&
MMU::slice(Tensor &t0, U32 x0, U32 x1, U32 y0, U32 y1) {
    if (t0.rank < 2) { ERROR("dim?"); return t0; }
    if (x1 == (U32)-1) x1 = t0.W();
    if (y1 == (U32)-1) y1 = t0.H();
    Tensor &t1 = t0.rank==2
        ? tensor(y1-y0, x1-x0)
        : tensor(t0.N(), y1-y0, x1-x0, t0.C());
    ///
    /// hard copy data blocks
    ///
    U32 N   = t1.N(), C = t1.C();
    U64 bsz = sizeof(DU) * C * t1.W();              /// size of one row
    for (int n = 0; n < N; n++) {                   /// repeat N HWC
        for (int j = y0, j0=0; j < y1; j++, j0++) {
            DU *d0 = &t0.data[C * (j * t0.W() + x0)];
            DU *d1 = &t1.data[C * j0 * t1.W()];
            memcpy(d1, d0, bsz);
        }
    }
    MM_DB("mmu#slice(T%d)[%d:%d,%d:%d,] numel=%ld\n",
          t0.rank, x0, x1, y0, y1, t0.numel);
    return t1;
}

#if T4_DO_NN
__HOST__ Dataset&                            ///< create a Dataset holder
MMU::dataset(U32 batch_sz) {                 /// * Note: data block is not allocated yet
    MM_DB("mmu#dataset batch_sz=%d {", batch_sz);
    Dataset *ds = (Dataset*)_mpool.malloc(); /// * was = (Dataset*)_ostore.malloc(sizeof(Dataset));
    ds->init(0, T4_DATASET, 4);
    ds->N()      = batch_sz;                 /// * other members filled in host mode
    ds->batch_id = 0;                        /// * setup control flag
    MM_DB("} mmu#dataset => D:%x\n", OBJ2X(*ds));
    return *ds;
}

__HOST__ Model&                              ///< create a NN model with NHWC input
MMU::model(int &trace, U32 sz) {
    MM_DB("mmu#model layers=%d {\n", sz);
    Model  *m = (Model*)_mpool.malloc();     /// * was = (Model*)_ostore.malloc(sizeof(Model));
    Tensor &t = talloc(sz);                  /// * allocate tensor storage
    m->init(this, t, trace);
    MM_DB("} mmu#model => M:%x\n", OBJ2X(*m));
    return *m;
}

__HOST__ void                                ///< release tensor memory blocks
MMU::free(Model &m) {
    int n = m.numel;
    MM_DB("mmu#free(N%d) N:%x {\n", n, OBJ2X(m));
    for (int i = m.numel-1; i >= 0; i--) {
        MM_DB("\t"); free(m[i]);             /// * release layer tensors
    }
    _mpool.free(&m);                         /// * release model itself
    MM_DB("} mmu#free(N%d)\n", n);
}
#endif // T4_DO_NN
#endif // T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

} // namespace t4::mu
