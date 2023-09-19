/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iomanip>       // setbase, setprecision
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
using namespace std;

#define NEXTNODE(n) ((obuf_node*)((char*)&node->data[0] + node->sz))

__HOST__ int
AIO::readline(std::istream &fin) {
    _istr->clear();
    char *tib = _istr->rdbuf();
    fin.getline(tib, T4_IBUF_SZ, '\n');
    return strlen(tib);
}

__HOST__ obuf_node*
AIO::process_node(std::ostream &fout, obuf_node *node) {
    cudaDeviceSynchronize();        /// * make sure data is completely written
    
    char *v = (char*)node->data;    ///< fetch payload in buffered print node
    switch (node->gt) {
    case GT_INT:   fout << (*(S32*)v); break;
    case GT_FLOAT: fout << (*(F32*)v); break;
    case GT_STR:   fout << v;          break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        //printf("FMT: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        fout << std::setbase(_radix = f->base)
             << std::setw(f->width)
             << std::setprecision(f->prec ? f->prec : -1)
             << std::setfill((char)f->fill);
    } break;
    case GT_OBJ: _print_obj(fout, *(DU*)v); break;
    case GT_OPX: {
        _opx *o = (_opx*)v;
        // printf("OP=%d a=%d, n=0x%08x=%f\n", o->op, o->a, DU2X(o->n), o->n);
        switch (o->op) {
        case OP_WORDS: _mmu->words(fout);                               break;
        case OP_SEE:   _mmu->see(fout, (IU)o->a);                       break;
        case OP_DUMP:  _mmu->mem_dump(fout, (IU)o->a, (IU)o->n);        break;
        case OP_SS:    _mmu->ss_dump(fout, (IU)node->id, o->a, _radix); break;
#if T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        case OP_TSAVE:
            node = NEXTNODE(node);
            _tsave(o->n, o->a, (char*)node->data);
            break;
#if T4_ENABLE_NN  //==========================================================
        case OP_DATA:
            node = NEXTNODE(node);                   ///< get dataset repo name
            _dsfetch(o->n, o->a, (char*)node->data); /// * fetch first batch
            break;
        case OP_FETCH: _dsfetch(o->n, o->a); break;  /// * fetch/rewind dataset batch
        case OP_NSAVE:
            node = NEXTNODE(node);                   ///< get dataset repo name
            _nsave(o->n, o->a, (char*)node->data);
            break;
        case OP_NLOAD:
            node = NEXTNODE(node);
            _nload(o->n, o->a, (char*)node->data);
            break;
#endif // T4_ENABLE_NN =======================================================
#endif // T4_ENABLE_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        }
    } break;
    default: fout << "print type not supported: " << (int)node->gt; break;
    }
    return NEXTNODE(node);
}

__HOST__ void
AIO::flush(std::ostream &fout) {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        node = process_node(fout, node);
    }
    _ostr->clear();
}
