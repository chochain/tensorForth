/**
 * @file
 * @brief tensorForth Async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
__HOST__ int
AIO::readline() {
    _istr->clear();
    char *tib = _istr->rdbuf();
    std::cin.getline(tib, T4_IBUF_SZ, '\n');
    if (_trace) std::cout << "<<" << tib << std::endl;
    return strlen(tib);
}

__HOST__ void
AIO::print_vec(DU *d, int mi, int ri) {
    std::cout << "[";
    for (int i=0;     i<ri; i++) std::cout << d[i] << ", ";
    for (int i=mi-ri; i<mi; i++) std::cout << d[i] << ", ";
    std::cout << "]";
}

__HOST__ void
AIO::print_mat(DU *d, int mi, int mj, int ri, int rj) {
    DU  *d0 = d, *d1 = d;
    for (int j=0;     j<rj; j++) { std::cout << "\n\t";  print_vec(d0++, mi, ri); }
    for (int j=mj-rj; j<mj; j++) { std::cout << ",\n\t"; print_vec(d1++, mi, ri); }
}

__HOST__ void
AIO::print_tensor(DU v) {
    Tensor &t = _mmu->du2ten(v);
    DU     *d = (DU*)t.data;
    printf("printing Tensor(%d,%d)=%p\n", t.shape[1], t.shape[2], &t);
    
    auto range = [this](int n) { return n > _edge ? _edge : n; };
    switch (t.rank) {
    case 1: {
        std::cout << "vector";
        int ri = range(t.size);
        print_vec(d, t.size, ri);
        std::cout << "\n";
    } break;
    case 2: {
        std::cout << "matrix[";
        int mj = t.shape[1], mi = t.shape[2], rj = range(mj),  ri = range(mi);
        print_mat(d, mi, mj, ri, rj);
        std::cout << "]\n";
    } break;
    case 4: {
        std::cout << "tensor["
                  << t.shape[2] << "," << t.shape[0] << "," << t.shape[1] << "," << t.shape[3]
                  << "]\n";
    } break;
    default: std::cout << "tensor rank=" << t.rank << " not supported\n";
    }
}

__HOST__ void
AIO::print_node(obuf_node *node) {
    if (_trace) std::cout << '<' << node->id << '>';

    char *v = (char*)node->data;
    switch (node->gt) {
    case GT_INT:   std::cout << (*(S32*)v); break;
    case GT_FLOAT: std::cout << (*(F32*)v); break;
    case GT_STR:   std::cout << v;          break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        //printf("FMT: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        std::cout << std::setbase(_radix = f->base)
                  << std::setw(f->width)
                  << std::setprecision(f->prec ? f->prec : -1)
                  << std::setfill((char)f->fill);
    } break;
    case GT_TENSOR: print_tensor(*(DU*)v); break;
    case GT_OPX: {
        OP  op = (OP)*v;
        U16 a  = *(U16*)(v+2);
        U16 n  = *(U16*)(v+4);
        //printf("OP=%d(%d, %d)\n", op, a, n);
        switch (op) {
        case OP_WORDS: _mmu->words(std::cout);              break;
        case OP_SEE:   _mmu->see(std::cout, (IU)a);         break;
        case OP_DUMP:  _mmu->mem_dump(std::cout, (IU)a, n); break;
        case OP_SS:    _mmu->ss_dump(std::cout, (IU)node->id, a, _radix); break;
        }
    } break;
    default: std::cout << "print type not supported: " << (int)node->gt;  break;
    }
    if (_trace) std::cout << "</" << node->id << '>' << std::endl;
}

#define NEXTNODE(n) ((obuf_node*)((char*)&node->data[0] + node->sz))
__HOST__ void
AIO::flush() {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        print_node(node);
        node = NEXTNODE(node);
    }
    _ostr->clear();
}
