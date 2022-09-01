/** -*- c++ -*-
 * @file
 * @brief tensorForth - Memory Manager
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>          // setw, setbase
#include "model.h"          // include mmu.h
///
/// random number generator setup
/// Note: kept here because curandStates stays in CUDA memory
///
__KERN__ void
k_rand_init(curandState *st, U64 seed=0) {
    ///
    /// serialize to make sure states are randomized
    ///
    for (U64 k = 0; k < T4_RAND_SZ; k++) {
        curand_init(seed != 0L ? seed : clock64(), k, k, &st[k]);
    }
}
__KERN__ void
k_rand(DU *mat, int sz, DU bias, DU scale, curandState *st, t4_rand_opt ntype) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    curandState *s = &st[threadIdx.x];

    if (k < sz) {
        mat[k] = scale * (
            bias + (ntype==NORMAL ? curand_normal(s) : curand_uniform(s))
        );
    }
}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
MMU::MMU(int verbose) : _trace(verbose) {
    cudaMallocManaged(&_dict, sizeof(Code) * T4_DICT_SZ);
    cudaMallocManaged(&_pmem, T4_PMEM_SZ);
    cudaMallocManaged(&_obj,  T4_TENSOR_SZ);
    cudaMallocManaged(&_mark, sizeof(DU) * T4_TFREE_SZ);
    cudaMallocManaged(&_vmss, sizeof(DU) * T4_SS_SZ * VM_MIN_COUNT);
    cudaMallocManaged(&_seed, sizeof(curandState) * T4_RAND_SZ);
    GPU_CHK();

    _ostore.init(_obj, T4_TENSOR_SZ);
    k_rand_init<<<1, T4_RAND_SZ>>>(_seed);
    GPU_CHK();
    
    TRACE1("\\  MMU dict=%p, mem=%p, vmss=%p, obj=%p\n", _dict, _pmem, _vmss, _obj);
}
__HOST__
MMU::~MMU() {
    GPU_SYNC();
    TRACE1("\\  MMU releasing CUDA managed memory...\n");
    cudaFree(_seed);
    cudaFree(_vmss);
    cudaFree(_mark);
    cudaFree(_obj);
    cudaFree(_pmem);
    cudaFree(_dict);
}
///
/// dictionary management methods
/// TODO: use const Code[] directly, as ROM, to prevent deep copy
///
__GPU__ int
MMU::find(const char *s, bool compile, bool ucase) {
    TRACE2("find(%s) => ", s);
    for (int i = _didx - (compile ? 2 : 1); i >= 0; --i) {
        const char *t = _dict[i].name;
        if (ucase && STRCASECMP(t, s)==0) return i;
        if (!ucase && STRCMP(t, s)==0) return i;
    }
    return -1;
}
__GPU__ void
MMU::merge(const Code *clist, int sz) {
    Code *c = (Code*)clist;
    for (int i=0; i<sz; i++, c++) {
        int w = find(c->name);              /// * check whether word exists
        if (w >= 0) {
            _dict[w] = *c;                  /// * replace existing word pointer
            TRACE1("word %s redefined\n", c->name);
        }
        else {
            add(c);                         /// * append new word to dictionary
            TRACE2("new word %s created\n", c->name);
        }
    }
}
__GPU__ void
MMU::status() {
    UFP x0 = ~0;                            ///< base of xt   allocations
    UFP n0 = ~0;                            ///< base of name allocations
    Code *c = _dict;
    for (int i=0; i<_didx; i++, c++) {      /// * scan thru for max range
        if ((UFP)c->xt   < x0) x0 = (UFP)c->xt;
        if ((UFP)c->name < n0) n0 = (UFP)c->name;
    }
    c = _dict;
    for (int i=0; i<_didx; i++, c++) {      ///< dump dictionary from device
        TRACE2("%3d> xt=%4x:%p name=%4x:%p %s\n", i,
            (U16)((UFP)c->xt   - x0), c->xt,
            (U16)((UFP)c->name - n0), c->name,
            c->name);
    }
    TRACE1("\\  MMU.alloc dict[%d/%d], pmem[%d]=%0.1f%%, tfree[%d/%d]\n",
        _didx, T4_DICT_SZ, _midx, 100.0*(_midx/T4_PMEM_SZ), _fidx, T4_TFREE_SZ);
}
///
/// colon - dictionary word compiler
///
__GPU__ void
MMU::colon(const char *name) {
    TRACE2("colon(%s) => ", name);
    int  sz = STRLENB(name);                // aligned string length
    Code &c = _dict[_didx++];               // get next dictionary slot
    align();                                // nfa 32-bit aligned (adjust _midx)
    c.name = (const char*)&_pmem[_midx];    // assign name field index
    c.def  = 1;                             // specify a colon word
    add((U8*)name,  ALIGN2(sz+1));          // setup raw name field
    c.pfa  = _midx;                         // capture code field index
}
#if T4_ENABLE_OBJ
///====================================================================
/// tensor life-cycle methods
///
__GPU__ void
MMU::mark_free(DU v) {            ///< mark a tensor free for release
    T4Base &t = du2obj(v);
    TRACE1("mark T[%x] as free[%d]\n", DU2X(v), _fidx);
//    lock();
    if (_fidx < T4_TFREE_SZ) _mark[_fidx++] = v;
    else ERROR("ERR: tfree store full, increase T4_TFREE_SZ!");
//    unlock();                   ///< TODO: CC: DEAD LOCK, now!
}
__GPU__ void                      ///< release marked free tensor
MMU::sweep() {
//    lock();
    for (int i = 0; _fidx && i < _fidx; i++) {
        DU v = _mark[i];
        TRACE1("release T[%x] from free[%d]\n", DU2X(v), i);
        drop(v);
    }
    _fidx = 0;
//  unlock();                      ///< TODO: CC: DEAD LOCK, now!
}
__GPU__ Tensor&                    ///< allocate a tensor from tensor space
MMU::talloc(U32 sz) {
    Tensor *t = (Tensor*)_ostore.malloc(sizeof(Tensor));
    void   *d = _ostore.malloc((U64)sizeof(DU) * sz);
    _ostore.status(_trace);
    t->reset(d, sz);
    return *t;
}
__GPU__ Tensor&                    ///< create a one-dimensional tensor
MMU::tensor(U32 sz) {
    TRACE1("mmu#tensor(%d) numel=%d", sz, sz);
    return talloc(sz);
}
__GPU__ Tensor&                    ///< create a 2-dimensional tensor
MMU::tensor(U16 h, U16 w) {
    U32 sz = h * w;
    TRACE1("mmu#tensor(%d,%d) numel=%d", h, w, sz);
    Tensor &t = talloc(sz);
    t.reshape(h, w);
    return t;
}
__GPU__ Tensor&                    ///< create a NHWC tensor
MMU::tensor(U16 n, U16 h, U16 w, U16 c) {
    U32 sz = n * h * w * c;
    TRACE1("mmu#tensor(%d,%d,%d,%d) numel=%d", n, h, w, c, sz);
    Tensor &t = talloc(sz);
    t.reshape(n, h, w, c);
    return t;
}
__GPU__ Tensor&                    ///< create a C1NHWC tensor
MMU::tensor(U16 c1, U16 n, U16 h, U16 w, U16 c) {
    U32 sz = c1 * n * h * w * c;
    TRACE1("mmu#tensor[%d](%d,%d,%d,%d) numel=%d", c1, n, h, w, c, sz);
    Tensor &t = talloc(sz);
    t.reshape(c1, n, h, w, c);
    return t;
}
__GPU__ Model&                     ///< create a NN model with NHWC input
MMU::model(U32 sz) {
    TRACE1("mmu#model layers=%d", sz);
    Model  *m = (Model*)_ostore.malloc(sizeof(Model));
    Tensor &t = talloc(sz);        /// * allocate tensor storage
    m->reset(this, t);
    return *m;
}
__GPU__ Tensor&                   ///< create a view of a Tensor
MMU::view(Tensor &t0) {
    if (t0.is_model()) return t0; ///> TODO: create model view
    Tensor *t = (Tensor*)_ostore.malloc(sizeof(Tensor));
    ///
    /// replicate A tensor
    ///
    memcpy(t, &t0, sizeof(Tensor));
    t->ttype = T4_VIEW;

    TRACE1("mmu#view => V%d numel=%d", t->rank, t->numel);
    _ostore.status(_trace);
    return *t;
}
__GPU__ void
MMU::resize(Tensor &t, U32 sz) {
    if (t.rank != 1) { ERROR("mmu#resize rank==1 only\n"); return; }
    TRACE1("mmu#resize numel=%d (was %d)", sz, t.numel);
    DU *d0 = t.data;             /// * keep original memory block
    t.data = (DU*)_ostore.malloc(sz * sizeof(DU));
    ///
    /// hardcopy tensor storage
    ///
    memcpy(t.data, d0, (t.numel < sz ? t.numel : sz) * sizeof(DU));
    t.shape[0] = t.numel = sz;   /// * adjust tensor storage size
    
    _ostore.free(d0);            /// * release 
    _ostore.status(_trace);
}
__GPU__ void                     ///< release tensor memory blocks
MMU::free(Tensor &t) {
    TRACE1("mmu#free(T%d) numel=%d", t.rank, t.numel);
    if (!t.is_view()) {          /// * skip view
        _ostore.free(t.data);    /// * free physical data
        if (t.grad_fn != L_NONE) {
            TRACE1(" {\n");
            for (int i=0; t.grad[i] && i < 4; i++) {
                TRACE1("\t\t"); free(*t.grad[i]);    /// recursive
            }
            TRACE1("\t}");
        }
    }
    _ostore.free(&t);            /// * free tensor object itself
    _ostore.status(_trace);
}
__GPU__ void                     ///< release tensor memory blocks
MMU::free(Model &m) {
    TRACE1("mmu#free(N%d) [\n", m.numel);
    for (int i = m.numel-1; i >= 0; i--) {
        TRACE1("\t"); free(m[i]);
    }
    TRACE1("]");
    _ostore.free(&m);
    _ostore.status(_trace);
}
///
/// deep copy a tensor
/// TODO: CDP
///
__GPU__ Tensor&
MMU::copy(Tensor &t0) {
    if (!t0.is_tensor()) return t0;    ///> skip, TODO: copy model

    Tensor *t1  = (Tensor*)_ostore.malloc(sizeof(Tensor));
    memcpy(t1, &t0, sizeof(Tensor));   /// * copy attributes
    t1->ttype = T4_TENSOR;             /// * physical copy, not a view
    ///
    /// hard copy data block
    ///
    U64 bsz = sizeof(DU) * t0.numel;
    t1->data = (DU*)_ostore.malloc(bsz);
    
    Tensor::copy(t0, *t1);
    DU off = obj2du(*t1);             /// * offset in object space
    TRACE1("mmu#copy(T%d) numel=%d to T[%x]", t0.rank, t0.numel, DU2X(off));
    _ostore.status(_trace);
    return *t1;
}
__GPU__ Tensor&
MMU::random(Tensor &t, t4_rand_opt ntype, DU bias, DU scale) {
    TRACE1("mmu#random(T%d) numel=%d bias=%.2f, scale=%.2f\n",
           t.rank, t.numel, bias, scale);
    dim3 blk(T4_RAND_SZ);
    dim3 grd((t.numel + blk.x - 1) / blk.x);
    
    k_rand<<<grd, blk>>>(t.data, t.numel, bias, scale, _seed, ntype);
    cudaDeviceSynchronize();

    return t;
}
///
/// short hands for eforth tensor ucodes (for DU <-> Tensor conversion)
/// TODO: more object types
///
__BOTH__ T4Base&
MMU::du2obj(DU d) {
    U32    *off = (U32*)&d;
    T4Base *t   = (T4Base*)(_obj + (*off & ~T4_OBJ_FLAG));
    return *t;
}
__BOTH__ DU
MMU::obj2du(T4Base &t) {
    U32 o = ((U32)((U8*)&t - _obj)) | T4_OBJ_FLAG;
    return *(DU*)&o;
}
__GPU__  void
MMU::drop(DU d) {
    if (!IS_OBJ(d)) return;
    T4Base &t = du2obj(d);
    bool m = t.is_model();
    if (t.is_model()) free((Model&)t);
    else              free((Tensor&)t);
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
    TRACE1("mmu#slice(T%d)[%d:%d,%d:%d,] numel=%d\n", t0.rank, t0.numel, x0, x1, y0, y1);
    return t1;
}
__GPU__ DU
MMU::rand(DU d, t4_rand_opt ntype) {
    if (!IS_OBJ(d)) return d * curand_uniform(&_seed[0]);
    random((Tensor&)du2obj(d), ntype);
    return d;
}
#endif // T4_ENABLE_OBJ
///
/// Debugging methods
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ int
MMU::to_s(std::ostream &fout, IU w) {
    /*
     * TODO: not sure why copying 32 byt does not work?
     * char name[36];
     * cudaMemcpy(name, _dict[w].name, 32, D2H);
     */
    Code &code = _dict[w];
    if (_trace) {
        fout << (code.immd ? "*" : " ")
             << "[" << std::setw(3) << w << "]"
             << code.xt << ":";
    }
    U8 c, i=0;
    cudaMemcpy(&c, code.name, 1, D2H);
    fout << " ";
    while (c) {
        fout << c;
        cudaMemcpy(&c, code.name+(++i), 1, D2H);
    }
    return (int)i;
}
__HOST__ int
MMU::to_s(std::ostream &fout, Tensor &t) {
    static const char tn[] = { 'V', 'T', 'N' };  /// sync with t4_obj
    auto t4 = [&fout, &t]() {
        fout << t.N() << "," << t.H() << "," << t.W() << "," << t.C() << "]";
    };
    fout << tn[t.ttype];
    switch(t.rank) {
    case 0: fout << "["  << (t.numel - 1) << "]";         break;
    case 1: fout << "1[" << t.numel << "]";               break;
    case 2: fout << "2[" << t.H() << "," << t.W() << "]"; break;
    case 4: fout << "4["; t4();                           break;
    case 5: fout << "5[" << t.parm << "]["; t4();         break;
    }
    return 1;
}
__HOST__ __INLINE__ int
MMU::to_s(std::ostream &fout, DU s) {
    return to_s(fout, (Tensor&)du2obj(s));
}
///
/// display dictionary word list
///
__HOST__ void
MMU::words(std::ostream &fout) {
    fout << std::setbase(10);
    for (int i=0, sz=0; i<_didx; i++) {
        sz += to_s(fout, (IU)i);
        if (_trace || sz > 54) { fout << std::endl; sz = 0; } /// TODO: width configuable
    }
    fout << std::endl;
}
///
/// recursively disassemble colon word
///
__HOST__ void
MMU::see(std::ostream &fout, U8 *ip, int dp) {
    while (*(IU*)ip) {                                              /// * loop until EXIT
        fout << std::endl; for (int n=dp; n>0; n--) fout << "  ";   /// * indentation by level
           fout << "[" << std::setw(4) << (IU)(ip - _pmem) << ": ";
        IU c = *(IU*)ip;                                            /// * fetch word index
        to_s(fout, c);                                              /// * display word name
        if (_dict[c].def && dp < 2) {                               /// * check if is a colon word
            see(fout, &_pmem[_dict[c].pfa], dp+1);                  /// * go one level deeper
        }
        ip += sizeof(IU);                                           /// * advance instruction pointer
        switch (c) {
        case DOVAR: case DOLIT: {                                   /// * fetch literal
            DU v = *(DU*)ip;  ip += sizeof(DU);
            fout << "= ";
            if (IS_OBJ(v)) to_s(fout, v);                           /// * handle object
            else fout << v;                                         /// * display the literal
        } break;
        case DOSTR: case DOTSTR: {
            char *s = (char*)ip;
            int  sz = strlen(s)+1;
            ip += ALIGN2(sz);                                       /// fetch string
            fout << "= \"" << s << "\"";
        } break;
        case BRAN: case ZBRAN: case DONEXT:
            fout << " j" << *(IU*)ip; ip += sizeof(IU); break;      /// fetch jump target
        }
        fout << "] ";
    }
}
__HOST__ void
MMU::see(std::ostream &fout, U16 w) {
    fout << "[ "; to_s(fout, w);
    if (_dict[w].def) see(fout, &_pmem[_dict[w].pfa]);
    fout << "]" << std::endl;
}
///
/// dump data stack content
///
__HOST__ void
MMU::ss_dump(std::ostream &fout, U16 vid, U16 n, int radix) {
    bool x = radix != 10;
    auto show = [this, &fout, x](DU s) {
        if (IS_OBJ(s)) to_s(fout, s);
        else if (x)    fout << static_cast<int>(s);
        else           fout << s;
    };
    DU *ss = &_vmss[vid * T4_SS_SZ];
    fout << " <";
    if (x) fout << std::setbase(radix);
    for (U16 i=0; i<n; i++) {
        show(ss[i]);
        fout << " ";
    }
    show(ss[T4_SS_SZ-1]);
    fout << "> ok" << std::endl;
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
MMU::mem_dump(std::ostream &fout, U16 p0, U16 sz) {
    static const char *i2h = "0123456789abcdef";
    char buf[80];
    for (U16 i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (U16 j=0; j<16; j++) {
            //U8 c = *(((U8*)&_dict[0])+i+j) & 0x7f;               // to dump _dict
            U8 c = _pmem[i+j];
            C2H(c);                                                // "%02x "
            c &= 0x7f;                                             // mask off high bit
            buf[x++] = ' ';
            if (j%4==3) buf[x++] = ' ';
            buf[59+j]= (c==0x7f||c<0x20) ? '.' : c;                // %c
        }
        buf[75] = '\0';
        fout << buf;
    }
    fout << std::endl;
}
