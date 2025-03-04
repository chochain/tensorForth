/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth System interface implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>
#include "sys.h"
#include "mmu/tensor.h"
#include "nn/dataset.h"
#include "nn/model.h"

System *_sys = NULL;
///
/// random number generator setup
/// Note: kept here because curandStates stays in CUDA memory
///
__KERN__ void
k_rand_init(curandState *st, U64 seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, x, 0, &st[x]);
}

__KERN__ void
k_rand(DU *mat, int sz, DU bias, DU scale, curandState *st, rand_opt ntype) {
    int tx = threadIdx.x;             ///< thread idx
    int n  = (sz / blockDim.x) + 1;   ///< loop counter
    
    curandState s = st[tx];           /// * cache state into local register
    for (int i=0, x=tx; i<n; i++, x+=blockDim.x) {  /// * scroll through pages
        if (x < sz) {
            mat[x]= scale * (
                bias + (ntype==NORMAL ? curand_normal(&s) : curand_uniform(&s))
                );
        }
    }
    st[tx] = s;                      /// * copy state back to global memory
}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
System::System(h_istr &i, h_ostr &o, int khz, int verbo)
    : _khz(khz), fin(i), _istr(new Istream()), _ostr(new Ostream()), _trace(verbo) {
    mu = MMU::get_mmu();             ///> instantiate memory controller
    io = AIO::get_io();              ///> instantiate async IO controler
    db = Debug::get_db(o, verbo);    ///> tracing instrumentation
    ///
    ///> setup randomizer
    ///
    MM_ALLOC(&_seed, sizeof(curandState) * T4_RAND_SZ);
    k_rand_init<<<1, T4_RAND_SZ>>>(_seed, time(NULL));  /// serialized randomizer
    GPU_CHK();
    
    INFO("\\ System OK\n");
}

__HOST__
System::~System() {
    GPU_SYNC();
    
    MM_FREE(_seed);
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

__GPU__ DU
System::rand(DU d, rand_opt n) {
    if (!IS_OBJ(d)) return d * curand_uniform(&_seed[threadIdx.x]);
#if T4_DO_OBJ
    T4Base &t = mu->du2obj(d);
    rand(t.data, t.numel, n);
    return d;
#endif // T4_DO_OBJ
}
__GPU__ void
System::rand(DU *d, U64 sz, rand_opt n, DU bias, DU scale) {
    DEBUG("sys#rand(T%d) numel=%ld bias=%.2f, scale=%.2f\n",
          t.rank, t.numel, bias, scale);
    k_rand<<<1, T4_RAND_SZ>>>(d, sz, bias, scale, _seed, n);
}
///
///> feed device input stream with a line from host input
///
#include <iostream>
#include <string>
__HOST__ int
System::readline() {
    _istr->clear();                          /// * clear device inpugt stream
    char *tib = _istr->rdbuf();              ///< set device input buffer
    fin.getline(tib, T4_IBUF_SZ, '\n');      /// * feed input from host to device
    return fin.eof();                        /// * end of input flag
}

#define NEXT_EVENT(n) ((io_event*)((char*)&ev->data[0] + ev->sz))

__HOST__ io_event*
System::process_event(io_event *ev) {
    GPU_SYNC();                     /// * make sure data is completely written

    void *v = (void*)ev->data;      ///< fetch payload in buffered print node
    switch (ev->gt) {
    case GT_INT:
    case GT_U32:
    case GT_FLOAT:
    case GT_STR:
    case GT_FMT:
    case GT_OBJ: db->print(v, ev->gt);                     break;
    case GT_OPX: {
        _opx *o = (_opx*)v;
        DEBUG("OP=%d, m=%d, i=%d, n=0x%08x=%g\n", o->op, o->m, o->i, DU2X(o->n), o->n);
        switch (o->op) {
        case OP_DICT:  db->dict_dump();                    break;
        case OP_WORDS: db->words();                        break;
        case OP_SEE:   db->see((IU)o->i, (int)o->m);       break;
        case OP_DUMP:  db->mem_dump((IU)o->i, UINT(o->n)); break;
        case OP_SS:    db->ss_dump((IU)o->i>>10, (int)o->i&0x3ff, o->n, (int)o->m); break;
#if T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        case OP_TSAVE: {
            Tensor &t = (Tensor&)mu->du2obj(o->n);
            if (t.is_tensor()) {
                ev = NEXT_EVENT(ev);
                io->tsave(t, (char*)ev->data, o->m);
            }
            else ERROR("%x is not a tensor\n", DU2X(o->n));
        } break;
#if T4_DO_NN  //==========================================================
        case OP_DATA: {
            Dataset &ds = (Dataset&)mu->du2obj(o->n);
            if (ds.is_dataset()) {                                         /// * indeed a dataset?
                ev = NEXT_EVENT(ev);                                            ///< get dataset repo name
                io->dsfetch(ds, (char*)ev->data, o->m); /// * fetch first batch
            }
            else ERROR("%x is not a dataset\n", DU2X(o->n));
        } break;
        case OP_FETCH: {
            Dataset &ds = (Dataset&)mu->du2obj(o->n);
            if (ds.is_dataset()) {
                io->dsfetch(ds, NULL, o->m);            /// * fetch/rewind dataset batch
            }
            else ERROR("%x is not a dataset\n", DU2X(o->n));
        } break;  
        case OP_NSAVE: {
            Model &m = (Model&)mu->du2obj(o->n);
            if (m.is_model()) {
                ev = NEXT_EVENT(ev);                                            ///< get dataset repo name
                io->nsave(m, (char*)ev->data, o->m);
            }
            else ERROR("%x is not a model\n", DU2X(o->n));
        } break;
        case OP_NLOAD: {
            Model &m = (Model&)mu->du2obj(o->n);
            if (m.is_model()) {
                ev = NEXT_EVENT(ev);
                _istr->clear();
                io->nload(m, (char*)ev->data, o->m, _istr->rdbuf());
            }
            else ERROR("%x is not a model\n", DU2X(o->n));
        } break;
#endif // T4_DO_NN =======================================================
#endif // T4_DO_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        }
    } break;
    default: ERROR("event type not supported: %d\n", (int)ev->gt); break;
    }
    return NEXT_EVENT(ev);
}

__HOST__ void
System::flush() {
    io_event *e = (io_event*)_ostr->rdbuf();
    while (e->gt != GT_EMPTY) {          // 0
        e = process_event(e);
    }
    _ostr->clear();
}


