/** -*- c++ -*-
 * @File
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
    if (_trace > 0) std::cout << "<<" << tib << std::endl;
    return strlen(tib);
}

__HOST__ void
AIO::print_vec(DU *d, int mi, int ri) {
    std::cout << "{";
    for (int i=0; i<ri; i++) {
        std::cout << " " << d[i];
    }
    int x = mi - ri;
    if (x > ri) std::cout << " ...";
    for (int i=(x > ri ? x : ri); i<mi; i++) {
        std::cout << " " << d[i];
    }
    std::cout << " }";
}

__HOST__ void
AIO::print_mat(DU *d, int mi, int mj, int ri, int rj) {
    bool full = (mi * mj) <= _thres;
    int  xi   = full ? mi : ri;
    DU   *d0  = d;
    for (int j=0, j1=1; j<rj; j++, j1++, d0+=mi) {
        print_vec(d0, mi, xi);
        std::cout << (j1==mj ? "" : "\n\t");
    }
    int y = full ? rj : mj - rj;
    if (y > rj) std::cout << "...\n\t";
    else y = rj;
    DU *d1 = d + y * mi;
    for (int j=y, j1=j+1; j<mj; j++, j1++, d1+=mi) {
        print_vec(d1, mi, xi);
        std::cout << (j1==mj ? "" : "\n\t");
    }
}

__HOST__ void
AIO::print_obj(DU v) {
    auto   range = [this](int n) { return (n < _edge) ? n : _edge; };
    
    Tensor &t = _mmu->du2ten(v);            ///< TODO: other object types
    DU     *d = (DU*)t.data;
    WARN("aio#print_tensor::T[%x]=%p data=%p\n", *(U32*)&v, &t, d);

    std::ios::fmtflags fmt0 = std::cout.flags();
    std::cout.flags(std::ios::showpos | std::ios::right | std::ios::fixed);
    std::cout << std::setprecision(_prec);
    switch (t.rank) {
    case 1: {
        std::cout << "array[" << t.size << "] = ";
        int ri = (t.size < _thres) ? t.size : range(t.size);
        print_vec(d, t.size, ri);
    } break;
    case 2: {
        std::cout << "matrix[" << t.H() << "," << t.W() << "] = {\n\t";
        int mj = t.H(), mi = t.W(), rj = range(mj),  ri = range(mi);
        print_mat(d, mi, mj, ri, rj);
        std::cout << " }";
    } break;
    case 4: {
        std::cout << "tensor["
                  << t.N() << "," << t.H() << "," << t.W() << "," << t.C()
                  << "]";
    } break;
    default: std::cout << "tensor rank=" << t.rank << " not supported";
    }
    std::cout << "\n";
    std::cout.flags(fmt0);
}

__HOST__ void
AIO::print_node(obuf_node *node) {
    cudaDeviceSynchronize();        /// * make sure data is completely written
    char *v = (char*)node->data;
    switch (node->gt) {
    case GT_INT:   std::cout << (*(I32*)v); break;
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
    case GT_OBJ: print_obj(*(DU*)v); break;
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
}

#define NEXTNODE(n) ((obuf_node*)((char*)&node->data[0] + node->sz))
__HOST__ void
AIO::flush() {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        if (_trace > 1) std::cout << '<' << node->id << '>';
        print_node(node);
        if (_trace > 1) std::cout << "</" << node->id << '>' << std::endl;
        node = NEXTNODE(node);
    }
    _ostr->clear();
}
