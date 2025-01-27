/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth System interface implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "sys.h"
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
k_rand(DU *mat, int sz, DU bias, DU scale, curandState *st, t4_rand_opt ntype) {
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
System::System() {
    MM_ALLOC(&_seed, sizeof(curandState) * T4_RAND_SZ);
    
    k_rand_init<<<1, T4_RAND_SZ>>>(_seed, time(NULL));  /// serialized randomizer
    GPU_CHK();
}

System::~System() {
    MM_FREE(_seed);
}

__HOST__ int
System::readline() {
    io->d_istr->clear();
    char *tib = io->d_istr->rdbuf();
    fin.getline(tib, T4_IBUF_SZ, '\n');
    return strlen(tib);
}

#define NEXT_EVENT(n) ((io_event*)((char*)&ev->data[0] + ev->sz))

__HOST__ io_event*
System::process_event(io_event *ev) {
    cudaDeviceSynchronize();        /// * make sure data is completely written
    
    char *v = (char*)ev->data;      ///< fetch payload in buffered print node
    switch (ev->gt) {
    case GT_INT:   fout << (*(S32*)v);                 break;
    case GT_U32:   fout << static_cast<U32>(*(U32*)v); break;
    case GT_FLOAT: fout << (*(F32*)v);                 break;
    case GT_STR:   fout << v;                          break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        //printf("FMT: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        fout << std::setbase(_radix = f->base)
             << std::setw(f->width)
             << std::setprecision(f->prec ? f->prec : -1)
             << std::setfill((char)f->fill);
    } break;
    case GT_OBJ: io->_print_obj(fout, *(DU*)v); break;
    case GT_OPX: {
        _opx *o = (_opx*)v;
        // printf("OP=%d a=%d, n=0x%08x=%f\n", o->op, o->a, DU2X(o->n), o->n);
        switch (o->op) {
        case OP_WORDS: dbg->words();                             break;
        case OP_SEE:   dbg->see((IU)o->a);                       break;
        case OP_DUMP:  dbg->mem_dump((IU)o->a, (IU)o->n);        break;
        case OP_SS:    dbg->ss_dump((IU)ev->id, o->a, _radix); break;
#if T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        case OP_TSAVE:
            ev = NEXT_EVENT(ev);
            io->_tsave(o->n, o->a, (char*)ev->data);
            break;
#if T4_ENABLE_NN  //==========================================================
        case OP_DATA:
            ev = NEXT_EVENT(ev);                         ///< get dataset repo name
            io->_dsfetch(o->n, o->a, (char*)ev->data);   /// * fetch first batch
            break;
        case OP_FETCH: io->_dsfetch(o->n, o->a); break;  /// * fetch/rewind dataset batch
        case OP_NSAVE:
            ev = NEXT_EVENT(ev);                         ///< get dataset repo name
            io->_nsave(o->n, o->a, (char*)ev->data);
            break;
        case OP_NLOAD:
            ev = NEXT_EVENT(ev);
            io->_nload(o->n, o->a, (char*)ev->data);
            break;
#endif // T4_ENABLE_NN =======================================================
#endif // T4_ENABLE_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        }
    } break;
    default: fout << "print type not supported: " << (int)ev->gt; break;
    }
    return NEXT_EVENT(ev);
}

__HOST__ void
System::flush(std::ostream &fout) {
    io_event *e = (io_event*)io->d_ostr->rdbuf();
    while (e->gt != GT_EMPTY) {          // 0
        e = process_event(fout, e);
    }
    io->d_ostr->clear();
}


