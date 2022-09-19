/** -*- c++ -*-
 * @File
 * @brief tensorForth Async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "model.h"
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
using namespace std;

__HOST__ int
AIO::readline() {
    _istr->clear();
    char *tib = _istr->rdbuf();
    cin.getline(tib, T4_IBUF_SZ, '\n');
    return strlen(tib);
}

__HOST__ void
AIO::print_node(obuf_node *node) {
    cudaDeviceSynchronize();        /// * make sure data is completely written
    char *v = (char*)node->data;
    switch (node->gt) {
    case GT_INT:   cout << (*(I32*)v); break;
    case GT_FLOAT: cout << (*(F32*)v); break;
    case GT_STR:   cout << v;          break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        //printf("FMT: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        cout << std::setbase(_radix = f->base)
             << std::setw(f->width)
             << std::setprecision(f->prec ? f->prec : -1)
             << std::setfill((char)f->fill);
    } break;
    case GT_OBJ: _print_obj(*(DU*)v); break;
    case GT_OPX: {
        _opx *o = (_opx*)v;
        // printf("OP=%d a=%d, n=0x%08x=%f\n", o->op, o->a, DU2X(o->n), o->n);
        switch (o->op) {
        case OP_WORDS: _mmu->words(cout);                               break;
        case OP_SEE:   _mmu->see(cout, (IU)o->a);                       break;
        case OP_DUMP:  _mmu->mem_dump(cout, (IU)o->a, (IU)o->n);        break;
        case OP_SS:    _mmu->ss_dump(cout, (IU)node->id, o->a, _radix); break;
        case OP_LOAD:  _mmu->load(cout, (IU)node->id, o->n);            break;
        }
    } break;
    default: cout << "print type not supported: " << (int)node->gt; break;
    }
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
///
/// private methods
///
#if T4_ENABLE_OBJ
__HOST__ void
AIO::_print_obj(DU v) {
    T4Base &b = _mmu->du2obj(v);
    switch (b.ttype) {
    case T4_VIEW:
    case T4_TENSOR: _print_tensor(v);
    case T4_MODEL:  _print_model(v);
    }
}
__HOST__ void
AIO::_print_vec(DU *d, int mi, int ri, int ci) {
    cout << "{";
    for (int i=0; i<ri; i++) {
        DU *dx = &d[i * ci];
        for (int c=0; c < ci; c++) {
            cout << (c>0 ? "_" : " ") << *dx++;
        }
    }
    int x = mi - ri;
    if (x > ri) cout << " ...";
    for (int i=(x > ri ? x : ri); i<mi; i++) {
        DU *dx = &d[i * ci];
        for (int c=0; c < ci; c++) {
            cout << (c>0 ? "_" : " ") << *dx++;
        }
    }
    cout << " }";
}
__HOST__ void
AIO::_print_mat(DU *d, int mi, int mj, int ri, int rj, int ci) {
    bool full = (mi * mj) <= _thres;
    int  xi   = full ? mi : ri;
    DU   *d0  = d;
    for (int j=0, j1=1; j<rj; j++, j1++, d0+=(mi * ci)) {
        _print_vec(d0, mi, xi, ci);
        cout << (j1==mj ? "" : "\n\t");
    }
    int y = full ? rj : mj - rj;
    if (y > rj) cout << "...\n\t";
    else y = rj;
    DU *d1 = (d + y * mi * ci);
    for (int j=y, j1=j+1; j<mj; j++, j1++, d1+=(mi * ci)) {
        _print_vec(d1, mi, xi, ci);
        cout << (j1==mj ? "" : "\n\t");
    }
}
__HOST__ void
AIO::_print_tensor(DU v) {
    auto   range = [this](int n) { return (n < _edge) ? n : _edge; };

    Tensor &t = (Tensor&)_mmu->du2obj(v);
    DU     *d = t.data;                     /// * short hand
    WARN("aio#print_tensor::T[%x]=%p data=%p\n", DU2X(v), &t, d);

    ios::fmtflags fmt0 = cout.flags();
    cout.flags(ios::showpos | ios::right | ios::fixed);
    cout << setprecision(_prec);
    switch (t.rank) {
    case 1: {
        cout << "vector[" << t.numel << "] = ";
        int ri = (t.numel < _thres) ? t.numel : range(t.numel);
        _print_vec(d, t.numel, ri, 1);
    } break;
    case 2: {
        cout << "matrix[" << t.H() << "," << t.W() << "] = {\n\t";
        int mj = t.H(), mi = t.W(), rj = range(mj),  ri = range(mi);
        _print_mat(d, mi, mj, ri, rj, 1);
        cout << " }";
    } break;
    case 4: {
        cout << "tensor["
             << t.N() << "," << t.H() << "," << t.W() << "," << t.C()
             << "] = {\n\t";
        int mj = t.H(), mi = t.W(), rj = range(mj),  ri = range(mi);
        _print_mat(d, mi, mj, ri, rj, t.C());
        cout << " }";
    } break;
    case 5: {
        cout << "tensor[" << t.parm << "]["
             << t.N() << "," << t.H() << "," << t.W() << "," << t.C()
             << "] = {...}";
    } break;        
    default: cout << "tensor rank=" << t.rank << " not supported";
    }
    cout << "\n";
    cout.flags(fmt0);
}
__HOST__ void
AIO::_print_model(DU v) {
    auto tinfo = [this](Tensor &t, int i, int fn) { ///> layer info
        cout << "[" << std::setw(3) << i << "] "
             << Model::nname(fn) << ":";
        _mmu->to_s(cout, t);
        int sz = t.grad[0] ? t.grad[0]->numel : 0;
        sz += t.grad[1] ? t.grad[1]->numel : 0;
        cout << ", #param=" << sz;
    };
    auto finfo = [this](Tensor **g) {
        for (int i=0; g[i] && i < 2; i++) {
            cout << " "; _mmu->to_s(cout, *g[i]);
        }
    };
    Model &m = (Model&)_mmu->du2obj(v);
    int   sz = m.numel;
    if (!m.is_model()) return;
    
    cout << "NN model[" << sz-1 << "/" << m.slots() << "]" << endl;
    for (int i = 1; i < sz; i++) {  /// skip root[0]
        Tensor &t = m[i];
        tinfo(t, i, (i==(sz-1)) ? 0 : t.grad_fn);
        if (_trace && t.grad_fn != L_NONE) finfo(t.grad);
        cout << endl;
    }
}
#endif // T4_ENABLE_OBJ
