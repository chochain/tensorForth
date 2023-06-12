/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "dataset.h"     // in ../mmu
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
using namespace std;
///
/// Tensor IO private methods
///
#if T4_ENABLE_OBJ
__HOST__ void
AIO::_print_obj(std::ostream &fout, DU v) {
    T4Base &b = _mmu->du2obj(v);
    switch (b.ttype) {
    case T4_VIEW:
    case T4_TENSOR:
    case T4_DATASET: _print_tensor(fout, v); break;
    case T4_MODEL:   _print_model(fout, v);  break;
    }
}
__HOST__ void
AIO::_print_vec(std::ostream &fout, DU *d, int mj, int rj, int c) {
    fout << setprecision(_prec) << "{";                 /// set precision
    for (int j=0; j<rj; j++) {
        DU *dx = &d[j * c];
        for (int k=0; k < c; k++) {
            fout << (k>0 ? "_" : " ") << *dx++;
        }
    }
    int x = mj - rj;
    if (x > rj) fout << " ...";
    for (int j=(x > rj ? x : rj); j<mj; j++) {
        DU *dx = &d[j * c];
        for (int k=0; k < c; k++) {
            fout << (k>0 ? "_" : " ") << *dx++;
        }
    }
    fout << " }";
}
__HOST__ void
AIO::_print_mat(std::ostream &fout, DU *d, int mi, int mj, int ri, int rj, int c) {
    fout.flags(ios::showpos | ios::right | ios::fixed); /// enforce +- sign
    bool full = (mi * mj) <= _thres;
    int  x    = full ? mj : rj;
    DU   *d0  = d;
    for (int i=0, i1=1; i<ri; i++, i1++, d0+=(mj * c)) {
        _print_vec(fout, d0, mj, x, c);
        fout << (i1==mi ? "" : "\n\t");
    }
    int y = full ? ri : mi - ri;
    if (y > ri) fout << "...\n\t";
    else y = ri;
    DU *d1 = (d + y * mj * c);
    for (int i=y, i1=i+1; i<mi; i++, i1++, d1+=(mj * c)) {
        _print_vec(fout, d1, mj, x, c);
        fout << (i1==mi ? "" : "\n\t");
    }
}
__HOST__ void
AIO::_print_tensor(std::ostream &fout, DU v) {
    auto range = [this](int n) { return (n < _edge) ? n : _edge; };

    Tensor &t = (Tensor&)_mmu->du2obj(v);
    DU     *d = t.data;                     /// * short hand
    WARN("aio#print_tensor::T[%x]=%p data=%p\n", DU2X(v), &t, d);

    ios::fmtflags fmt0 = fout.flags();
    fout << setprecision(-1);               /// * standard format
    switch (t.rank) {
    case 1: {
        fout << "vector[" << t.numel << "] = ";
        int ri = (t.numel < _thres) ? t.numel : range(t.numel);
        _print_vec(fout, d, t.numel, ri, 1);
    } break;
    case 2: {
        int mi = t.H(), mj = t.W(), ri = range(mi),  rj = range(mj);
        fout << "matrix[" << mi << "," << mj << "] = {\n\t";
        _print_mat(fout, d, mi, mj, ri, rj, 1);
        fout << " }";
    } break;
    case 4: {
        int n  = t.N(), mi = t.H(), mj = t.W(), mc = t.C();
        int ri = range(mi), rj = range(mj);
        int pg = mi * mj * mc;
        fout << "tensor["
             << n << "," << mi << "," << mj << "," << mc
             << "] = {\n\t";
        for (int i = 0; i < n; i++, d += pg) {
            if (mj==1) _print_mat(fout, d, mj, mi, rj, ri, mc);
            else       _print_mat(fout, d, mi, mj, ri, rj, mc);
            fout << ((i+1) < n ? "\n\t" : "");
        }
        fout << " }";
    } break;
    case 5: {
        fout << "tensor[" << t.parm << "]["
             << t.N() << "," << t.H() << "," << t.W() << "," << t.C()
             << "] = {...}";
    } break;        
    default: fout << "tensor rank=" << t.rank << " not supported";
    }
    fout << "\n";
    fout.flags(fmt0);
}
///
/// Tensor & NN model persistence (i.e. serialization) methods
///
#include <fstream>
__HOST__ int
AIO::_tsave(DU top, bool raw, char *fname) {
    printf("\nAIO::save tensor to '%s' =>", fname);
    Tensor &t = (Tensor&)_mmu->du2obj(top);
    ofstream fout(fname, ios_base::binary);     ///< open an output file
    if (!fout.is_open()) {
        ERROR(" failed to open for output\n");
        return 1;
    }
    if (raw) _tsave_raw(fout, t);               ///< write in NHWC byte format
    else     _tsave_npy(fout, t);               ///< write in Numpy format
    fout.close();
    printf(" completed\n");
    return 0;
}

__HOST__ int
AIO::_tsave_raw(std::ostream &fout, Tensor &t) {
    const int N = t.N(), sz = t.HWC();
    char *buf = (char*)malloc(sz);
    for (int n=0; n < N; n++) {
        for (int i=0; i < sz; i++) {
            buf[i] = static_cast<U8>(t.data[i + n * N]);
        }
        fout.write(buf, sz);
    }
    free(buf);
    return 0;
}
__HOST__ int
AIO::_tsave_npy(std::ostream &fout, Tensor &t) {
    /// TODO:
    return 0;
}
#endif // T4_ENABLE_OBJ
