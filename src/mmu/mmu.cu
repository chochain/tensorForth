/** -*- c++ -*-
 * @file
 * @brief MMU class - memory manager implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>             // setw, setbase
#include "mmu.h"

///@name static class member
///@note: CUDA does not support device static data
///@{
__GPU__  UFP _XT0;
__GPU__  UFP _NM0;
///@}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
MMU::MMU() {
    MM_ALLOC(&_dict, sizeof(Code) * T4_DICT_SZ);
    MM_ALLOC(&_vmss, sizeof(DU) * T4_SS_SZ * T4_VM_COUNT);
    MM_ALLOC(&_vmrs, sizeof(DU) * T4_RS_SZ * T4_VM_COUNT);
    MM_ALLOC(&_pmem, T4_PMEM_SZ);
    
#if T4_ENABLE_OBJ    
    MM_ALLOC(&_mark, sizeof(DU) * T4_TFREE_SZ);
    MM_ALLOC(&_obj,  T4_OSTORE_SZ);
    _ostore.init(_obj, T4_OSTORE_SZ);
#endif // T4_ENABLE_OBJ

    _midx = T4_USER_AREA;      // set aside user area (for base and maybe compile)
    
    TRACE(
        "\\ MMU: CUDA Managed Memory\n"
        "\\\tdict=%p\n"
        "\\\tvmss=%p\n"
        "\\\tvmrs=%p\n"
        "\\\tmem =%p\n"
        "\\\tmark=%p\n"
        "\\\tobj =%p\n",
        _dict, _vmss, _vmrs, _pmem, _mark, _obj);
}
__HOST__
MMU::~MMU() {
    if (_obj)  MM_FREE(_obj);
    if (_mark) MM_FREE(_mark);
    MM_FREE(_pmem);
    MM_FREE(_vmrs);
    MM_FREE(_vmss);
    MM_FREE(_dict);
    TRACE("\\ MMU: CUDA Managed Memory freed\n");
}
///
/// static functions (for type conversion)
///
__GPU__  FPTR MMU::XT(IU ioff)      { return (FPTR)(_XT0 + ioff);  }
__GPU__  IU   MMU::XTOFF(FPTR xt)   { return (IU)((UFP)xt - _XT0); }
///
/// dictionary management methods
/// TODO: use const Code[] directly, as ROM, to prevent deep copy
///
__GPU__ void
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

__GPU__ IU
MMU::find(const char *s) {
    IU v = 0;
    DEBUG("mmu.find(%s) => ", s);
    for (IU i = _didx - 1; _didx && !v && i > 0; --i) {
        if (STRCMP(_dict[i].name, s)==0) v = i;
    }
    return v;
}

__GPU__ void
MMU::status() {
    INFO("\\ MMU.stat dict[%d/%d], pmem[%d]=%0.1f%%, tfree[%d/%d]\n",
        _didx, T4_DICT_SZ, _midx, 100.0*(_midx/T4_PMEM_SZ), _fidx, T4_TFREE_SZ);
    ///
    /// display object store statistics
    ///
#if T4_ENABLE_OBJ    
    _ostore.status(_trace);
#endif // T4_ENABLE_OBJ
}

__GPU__ void
MMU::dict_dump() {
    Code *c = _dict;
    DEBUG("Built-in Dictionary [name0=0x%lx, xt0=0x%lx]\n", _NM0, _XT0);
    for (int i=0; i<_didx; i++, c++) {      ///< dump dictionary from device
        IU  ix = c->udf ? c->pfa : (U32)(((UFP)c->xt & MSK_XT) - _XT0);
        U32 sz = ALIGN(STRLEN(c->name) + 1);
        DEBUG("%4d|%03x> name=%6x, %s=%6x %s\n", i, i,
              c->udf ? (c->pfa - sz) : (U32)((UFP)c->name - _NM0),
              c->udf ? "pf" : "xt", ix, c->name);
    }
}
///
/// colon - dictionary word compiler
///
__GPU__ void
MMU::colon(const char *name) {
    DEBUG("colon(%s) => ", name);
    int  sz = STRLENB(name);                // aligned string length
    Code &c = _dict[_didx++];               // get next dictionary slot
    align();                                // nfa 32-bit aligned (adjust _midx)
    c.didx = _didx-1;                       // directory index (reverse link)
    c.nfa  = _midx;                         // name field offset
    c.name = (const char*)&_pmem[_midx];    // assign name field index
    c.udf  = 1;                             // specify a colon word
    add((U8*)name,  ALIGN(sz+1));           // setup raw name field
    c.pfa  = _midx;                         // parameter field offset
}
///
/// tensor life-cycle methods
///
#if T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#include "model.h"             // in ../nn, include ../mmu/mmu.h
#include "dataset.h"           // in ../nn

#define OBJ2X(t)  ((U32)((U8*)&(t) - _obj))
__GPU__ void
MMU::mark_free(DU v) {            ///< mark a tensor free for release
    if (IS_VIEW(v)) return;
    T4Base &t = TBase::du2obj(v);
    DEBUG("mmu#mark T[%x] to free[%d]\n", OBJ2X(t), _fidx);
//    lock();
    if (_fidx < T4_TFREE_SZ) _mark[_fidx++] = T4base::obj2du(t);
    else ERROR("ERR: tfree store full, increase T4_TFREE_SZ!");
//    unlock();                   ///< TODO: CC: DEAD LOCK, now!
}
__GPU__ void                      ///< release marked free tensor
MMU::sweep() {
//    lock();
    for (int i = 0; _fidx && i < _fidx; i++) {
        DU v = _mark[i];
        DEBUG("mmu#release T[%x] from free[%d]\n", DU2X(v) & ~T4_TT_OBJ, i);
        drop(v);
    }
    _fidx = 0;
//  unlock();                      ///< TODO: CC: DEAD LOCK, now!
}
__GPU__ Tensor&                    ///< allocate a tensor from tensor space
MMU::talloc(U32 sz) {
    Tensor &t = *(Tensor*)_ostore.malloc(sizeof(Tensor));
    DEBUG(" T[%x]", OBJ2X(t));
    void   *d = _ostore.malloc((U64)sizeof(DU) * sz);
    _ostore.status(_trace);
    t.reset(d, sz);
    return t;
}
__GPU__ Tensor&                    ///< create a one-dimensional tensor
MMU::tensor(U32 sz) {
    DEBUG("mmu#tensor(%d) numel=%d", sz, sz);
    return talloc(sz);
}
__GPU__ Tensor&                    ///< create a 2-dimensional tensor
MMU::tensor(U16 h, U16 w) {
    U32 sz = h * w;
    DEBUG("mmu#tensor(%d,%d) numel=%d", h, w, sz);
    Tensor &t = talloc(sz);
    t.reshape(h, w);
    return t;
}
__GPU__ Tensor&                    ///< create a NHWC tensor
MMU::tensor(U16 n, U16 h, U16 w, U16 c) {
    U32 sz = n * h * w * c;
    DEBUG("mmu#tensor(%d,%d,%d,%d) numel=%d", n, h, w, c, sz);
    Tensor &t = talloc(sz);
    t.reshape(n, h, w, c);
    return t;
}
__GPU__ void
MMU::resize(Tensor &t, U32 sz) {
    if (t.rank != 1) { ERROR("mmu#resize rank==1 only\n"); return; }
    DEBUG("mmu#resize numel=%d (was %d)", sz, t.numel);
    DU *d0 = t.data;             /// * keep original memory block
    t.data = (DU*)_ostore.malloc(sz * sizeof(DU));
    ///
    /// hardcopy tensor storage
    ///
    memcpy(t.data, d0, (t.numel < sz ? t.numel : sz) * sizeof(DU));
    t.H() = t.numel = sz;        /// * adjust tensor storage size
    
    _ostore.free(d0);            /// * release 
    _ostore.status(_trace);
}
__GPU__ void                     ///< release tensor memory blocks
MMU::free(Tensor &t) {
    DEBUG("mmu#free(T%d) numel=%d T[%x]", t.rank, t.numel, OBJ2X(t));
    _ostore.free(t.data);        /// * free physical data
    if (t.grad_fn != L_NONE) {
        DEBUG(" {\n");
        for (int i=0; t.mtum[i] && i < 4; i++) {
            if (t.mtum[i] == t.grad[i]) continue;   /// * dummy pointers for SGD
            DEBUG("\t\t"); free(*t.mtum[i]);
        }
        for (int i=0; t.grad[i] && i < 4; i++) {
            DEBUG("\t\t"); free(*t.grad[i]);    /// recursive
        }
        DEBUG("\t}");
    }
    _ostore.free(&t);              /// * free tensor object itself
    _ostore.status(_trace);
}
#if T4_ENABLE_NN
__GPU__ Model&                     ///< create a NN model with NHWC input
MMU::model(U32 sz) {
    DEBUG("mmu#model layers=%d", sz);
    Model  *m = (Model*)_ostore.malloc(sizeof(Model));
    Tensor &t = talloc(sz);        /// * allocate tensor storage
    m->reset(this, t);
    return *m;
}
__GPU__ Dataset&                   ///< create a Dataset holder
MMU::dataset(U16 batch_sz) {       /// * Note: data block is not allocated yet
    DEBUG("mmu#dataset batch_sz=%d", batch_sz);
    Dataset *ds = (Dataset*)_ostore.malloc(sizeof(Dataset));
    ds->init(0, T4_DATASET, 4);
    ds->N()      = batch_sz;       /// * other members filled in host mode
    ds->batch_id = 0;              /// * setup control flag
    _ostore.status(_trace);
    return *ds;
}
__GPU__ void                     ///< release tensor memory blocks
MMU::free(Model &m) {
    DEBUG("mmu#free(N%d) [\n", m.numel);
    for (int i = m.numel-1; i >= 0; i--) {
        DEBUG("\t"); free(m[i]);
    }
    DEBUG("]");
    _ostore.free(&m);
    _ostore.status(_trace);
}
#endif // T4_ENABLE_NN
///
/// deep copy a tensor
/// TODO: CDP
///
__GPU__ Tensor&
MMU::copy(Tensor &t0) {
    if (!t0.is_tensor()) return t0;    ///> skip, TODO: copy model

    Tensor &t1  = *(Tensor*)_ostore.malloc(sizeof(Tensor));
    memcpy(&t1, &t0, sizeof(Tensor));   /// * copy attributes
    ///
    /// set attributes
    ///
    for (int i=0; i<4; i++) t1.grad[i] = t1.mtum[i] = NULL;  /// * blank gradients
    t1.grad_fn = L_NONE;                /// * not a network layer
    t1.nref    = 1;                     /// * reset ref counter
    ///
    /// hard copy data block
    ///
    U64 bsz = sizeof(DU) * t0.numel;
    t1.data = (DU*)_ostore.malloc(bsz);
    t1 = t0;                            /// * copy all tensor elements
    
    DBUG("mmu#copy(T%d) numel=%d to T[%x]", t0.rank, t0.numel, OBJ2X(t1));
    _ostore.status(_trace);
    
    return t1;
}
__GPU__ Tensor&
MMU::random(Tensor &t, t4_rand_opt ntype, DU bias, DU scale) {
    DEBUG("mmu#random(T%d) numel=%d bias=%.2f, scale=%.2f\n",
              t.rank, t.numel, bias, scale);
    k_rand<<<1, T4_RAND_SZ>>>(t.data, t.numel, bias, scale, _seed, ntype);
    GPU_SYNC();
    
    return t;
}
///
/// tensor slice & dice
/// TODO: CDP
///
__GPU__ Tensor&
MMU::slice(Tensor &t0, U16 x0, U16 x1, U16 y0, U16 y1) {
    if (t0.rank < 2) { ERROR("dim?"); return t0; }
    if (x1 == (U16)-1) x1 = t0.W();
    if (y1 == (U16)-1) y1 = t0.H();
    Tensor &t1 = t0.rank==2
        ? tensor(y1-y0, x1-x0)
        : tensor(t0.N(), y1-y0, x1-x0, t0.C());
    ///
    /// hard copy data blocks
    ///
    U16 N   = t1.N(), C = t1.C();
    U64 bsz = sizeof(DU) * C * t1.W();              // size of one row
    for (int n = 0; n < N; n++) {                   // repeat N HWC
        for (int j = y0, j0=0; j < y1; j++, j0++) {
            DU *d0 = &t0.data[C * (j * t0.W() + x0)];
            DU *d1 = &t1.data[C * j0 * t1.W()];
            memcpy(d1, d0, bsz);
        }
    }
    DEBUG("mmu#slice(T%d)[%d:%d,%d:%d,] numel=%d\n",
              t0.rank, t0.numel, x0, x1, y0, y1);
    return t1;
}
#endif // T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
