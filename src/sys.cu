/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth System interface implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>
#include "sys.h"                     /// include mmu/tensor.h
#include "nn/dataset.h"
#include "nn/model.h"

System  *_sys = NULL;                ///< singleton controller on host
__GPU__ curandState *_rand_st;       ///< for random number generator
__GPU__ int _khz  = 0;
///
/// random number generator setup
/// Note: kept here because curandStates stays in CUDA memory
///
__KERN__ void
k_rand_init(U64 seed, int khz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x==0) {
        _khz = khz;
        _rand_st = (curandState*)malloc(sizeof(curandState) * T4_RAND_SZ);
    }
    __syncthreads();
    curand_init(seed, x, 0, &_rand_st[x]);
}

__KERN__ void
k_rand(DU *mat, U64 sz, DU bias, DU scale, rand_opt ntype) {
    U32 n  = (sz / blockDim.x) + 1;  ///< loop counter
    U32 tx = threadIdx.x;            ///< thread idx (T4_RAN_SZ)
    U64 x  = (U64)tx;                 
    
    curandState s = _rand_st[tx];    /// * cache state into local register
    for (U32 i=0; i<n; i++, x+=blockDim.x) {  /// * scroll through pages
        if (x < sz) {
            mat[x]= scale * (
                bias + (ntype==NORMAL ? curand_normal(&s) : curand_uniform(&s))
                );
        }
    }
    _rand_st[tx] = s;                /// * copy state back to global memory
}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
System::System(h_istr &i, h_ostr &o, int khz, int verbo)
    : fin(i), fout(o), _istr(new Istream()), _ostr(new Ostream()), _trace(verbo) {
    mu = MMU::get_mmu();             ///> instantiate memory controller
    io = AIO::get_io(&_trace);       ///> instantiate async IO controler
    db = Debug::get_db(o);           ///> tracing instrumentation
    ///
    ///> setup randomizer
    ///
//    MM_ALLOC(&_rand_st, sizeof(curandState) * T4_RAND_SZ);
    k_rand_init<<<1, T4_RAND_SZ>>>(time(NULL), khz);  /// serialized randomizer
    GPU_CHK();

    INFO("\\ System OK\n");
}

__HOST__
System::~System() {
    GPU_SYNC();

//    MM_FREE(_rand_st);
    AIO::free_io();
    Debug::free_db();
    MMU::free_mmu();
    INFO("\\ System: instance freed\n");
}

__HOST__ System*
System::get_sys(h_istr &i, h_ostr &o, int khz, int verbo) {
    if (!_sys) _sys = new System(i, o, khz, verbo);
    return _sys;
}
__HOST__ System *System::get_sys()  { return _sys; }
__HOST__ void    System::free_sys() { if (_sys) delete _sys; }
///
///@name cross platform timing support
///
__GPU__ DU
System::ms() { return static_cast<double>(clock64()) / _khz; }

__GPU__ void
System::delay(int ticks) {
    U64 t = clock() + (ticks * _khz);
    while ((U64)clock() < t) { /* spinning */ };
}

__GPU__ void
System::rand(DU *d, U64 sz, rand_opt n, DU bias, DU scale) {
    /// rand states are dependent, cannot run parallel with multi-blocks
    k_rand<<<1, T4_RAND_SZ>>>(d, sz, bias, scale, n);
}

__GPU__ DU
System::rand(DU d, rand_opt n) {
    if (!IS_OBJ(d)) return d * curand_uniform(&_rand_st[threadIdx.x]);
#if T4_DO_OBJ
    T4Base &t = mu->du2obj(d);
    rand(t.data, t.numel, n);
    return d;
#endif // T4_DO_OBJ
}
///@}
///@name event loop handler
///@{
#include <iostream>
#include <string>
__HOST__ int
System::readline(int hold) {                 ///< feed a line into device input stream
    if (hold) return 1;                      ///< do not clear input buffer
    _istr->clear();                          /// * clear device inpugt stream
    char *tib = _istr->rdbuf();              ///< set device input buffer
    fin.getline(tib, T4_IBUF_SZ, '\n');      /// * feed input from host to device
    return !fin.eof();                       /// * more to read?
}

#define NEXT_EVENT(n) ((io_event*)((char*)&ev->data[0] + ev->sz))

__HOST__ io_event*
System::process_event(io_event *ev) {
    GPU_SYNC();                     /// * make sure data is completely written

    void *v = (void*)ev->data;      ///< fetch payload in buffered print node
    DEBUG("System::process(gt=%x) {\n", ev->gt);
    switch (ev->gt) {
    ///> simple ops
    case GT_INT:
    case GT_U32:
    case GT_FLOAT:
    case GT_STR:
    case GT_FMT:
    case GT_OBJ: db->print(v, ev->gt);                       break;
    ///> complex ops
    case GT_OPX: {
        _opx *o = (_opx*)v;
        DEBUG("  _opx(OP=%d, m=%d, i=%d, n=0x%08x=%g)\n", o->op, o->m, o->i, DU2X(o->n), o->n);
        switch (o->op) {
        case OP_FLUSH: fout << std::flush;                   break;
        case OP_DICT:  db->dict_dump();                      break;
        case OP_WORDS: db->words();                          break;
        case OP_SEE:   db->see(o->i, o->m);                  break;
        case OP_DUMP:  db->mem_dump(UINT(o->n), o->i);       break;
        case OP_SS:    db->ss_dump(o->n, o->i, o->m);        break;
#if T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        case OP_TSAVE: {
            Tensor &t = (Tensor&)mu->du2obj(o->n);
            if (t.is_tensor()) {
                ev = NEXT_EVENT(ev);
                char *fn = (char*)ev->data;                     ///> filename
                io->tsave(t, fn, o->m);                         /// * save tensor
            }
            else ERROR("%x is not a tensor\n", DU2X(o->n));
        } break;
#if T4_DO_NN  //==========================================================
        case OP_DATA: {
            Dataset &ds = (Dataset&)mu->du2obj(o->n);
            if (ds.is_dataset()) {                              /// * indeed a dataset?
                ev = NEXT_EVENT(ev);                            ///< get dataset repo name
                char *ds_nm = (char*)ev->data;                  /// * dataset name
                io->dsfetch(ds, ds_nm, 0);                      /// * fetch first batch
            }
            else ERROR("%x is not a dataset\n", DU2X(o->n));
        } break;
        case OP_FETCH: {
            Dataset &ds = (Dataset&)mu->du2obj(o->n);
            if (ds.is_dataset()) {
                io->dsfetch(ds, NULL, o->m);                    /// * fetch/rewind dataset batch
            }
            else ERROR("%x is not a dataset\n", DU2X(o->n));
        } break;  
        case OP_NSAVE: {
            printf("OP_NSAVE %x\n", DU2X(o->n));
            Model &m = (Model&)mu->du2obj(o->n);
            if (m.is_model()) {
                ev = NEXT_EVENT(ev);                            ///< get dataset repo name
                char *fn = (char*)ev->data;                     ///< filename
                io->nsave(m, fn, o->m);                         /// * o->m FAM mode
            }
            else ERROR("%x is not a model\n", DU2X(o->n));
        } break;
        case OP_NLOAD: {
            Model &m = (Model&)mu->du2obj(o->n);
            if (m.is_model()) {
                ev = NEXT_EVENT(ev);
                _istr->clear();
                char *fn = (char*)ev->data;                    ///< filename
                io->nload(m, fn, o->m, _istr->rdbuf());        /// * fetch into rdbuf
            }
            else ERROR("%x is not a model\n", DU2X(o->n));
        } break;
#endif // T4_DO_NN =======================================================
#endif // T4_DO_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        }
    } break;
    default: ERROR("event type not supported: %d\n", (int)ev->gt); break;
    }
    DEBUG("} System::process(gt=%x)\n", ev->gt);
    return NEXT_EVENT(ev);
}

__HOST__ void
System::flush() {
    io_event *e = (io_event*)_ostr->rdbuf();
    while (e->gt != GT_EMPTY) {          // walk linked-list
        e = process_event(e);
    }
    fout << std::flush;
    _ostr->clear();
}
///@}


