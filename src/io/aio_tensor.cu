/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iomanip>       // setbase, setprecision
#include "aio.h"

#if T4_ENABLE_OBJ
///
/// Tensor IO private methods
///
__HOST__ void
AIO::_print_vec(DU *vd, int W, int C) {
    int rw = (W <= _thres) ? W : (W < _edge ? W : _edge);
    fout << setprecision(_prec) << "{";                 /// set precision
    for (int j=0; j < rw; j++) {
        DU *dx = vd + j * C;
        for (int k=0; k < C; k++) {
            fout << (k>0 ? "_" : " ") << *dx++;
        }
    }
    int x = W - rw;
    if (x > rw) fout << " ...";
    for (int j=(x > rw ? x : rw); j < W; j++) {
        DU *dx = vd + j * C;
        for (int k=0; k < C; k++) {
            fout << (k>0 ? "_" : " ") << *dx++;
        }
    }
    fout << " }";
}
__HOST__ void
AIO::_print_mat(DU *td, U16 *shape) {
    auto range = [this](int v) { return (v < _edge) ? v : _edge; };
    const int H = shape[0], W = shape[1], C = shape[2]; ///< height, width, channels
    const int rh= range(H), rw=range(W);                ///< h,w range for ...
    DU *d = td;
    
    fout.flags(ios::showpos | ios::right | ios::fixed); /// enforce +- sign
    for (int y=0, y1=1; y<rh; y++, y1++, d+=(W * C)) {
        _print_vec(fout, d, W, C);
        fout << (y1==H ? "" : "\n\t");
    }

    int ym = (H <= _thres) ? rh : H - rh;
    if (ym > rh) fout << "...\n\t";
    else ym = rh;
    
    d = td + ym * W * C;
    for (int y=ym, y1=y+1; y<H; y++, y1++, d+=(W * C)) {
        _print_vec(fout, d, W, C);
        fout << (y1==H ? "" : "\n\t");
    }
}
__HOST__ void
AIO::_print_tensor(Tensor &t) {
    DU *td = t.data;                        /// * short hand
    DEBUG("aio#print_tensor::T=%p data=%p\n", &t, td);

    ios::fmtflags fmt0 = fout.flags();
    fout << setprecision(-1);               /// * standard format
    switch (t.rank) {
    case 1: {
        fout << "vector[" << t.numel << "] = ";
        _print_vec(fout, td, t.numel, 1);
    } break;
    case 2: {
        fout << "matrix[" << t.H() << "," << t.W() << "] = {\n\t";
        _print_mat(fout, td, t.shape);
        fout << " }";
    } break;
    case 4: {
        int N = t.N();
        fout << "tensor["
             << N << "," << t.H() << "," << t.W() << "," << t.C()
             << "] = { {\n\t";
        for (int n = 0; n < N; n++, td += t.HWC()) {
            _print_mat(fout, td, t.shape);
            fout << ((n+1) < N ? " } {\n\t" : "");
        }
        fout << " } }";
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
AIO::_tsave(DU top, U16 mode, char *fname) {
    printf("\nAIO::save tensor to '%s' =>", fname);
    
    ios_base::openmode m = (mode & FAM_RW) ? ios_base::in : ios_base::out;
    if (mode & FAM_RAW) m |= ios_base::binary;
    
    Tensor &t = (Tensor&)_mmu->du2obj(top);
    ofstream fout(fname, m);                      ///< open an output file
    if (!fout.is_open()) {
        ERROR(" failed to open for output\n");
        return 1;
    }
    if (mode & FAM_RAW) _tsave_raw(fout, t);      /// * write in raw format
    else                _tsave_txt(fout, t);      /// * write in text format
    
    fout.close();
    printf(" completed\n");
    return 0;
}

__HOST__ int
AIO::_tsave_txt(h_ostr &fout, Tensor &t) {
    int tmp = _thres;
    _thres  = 1024;                              /// * allow 1K*1K cells
    _print_tensor(fout, t);              
    _thres  = tmp;
    return 0;
}

__HOST__ int
AIO::_tsave_raw(h_ostr &fout, Tensor &t) {
    const char hdr[2] = { 'T', '4' };
    const int N = t.N(), HWC = t.HWC();
    U8 *buf = (U8*)malloc(HWC);                       ///< buffer for one slice

    fout.write(hdr, sizeof(hdr));
    fout.write((const char*)t.shape, sizeof(t.shape));
    for (int n=0; n < N; n++) {
        DU *p = &t.data[n * HWC];                    ///< slice-by-slice
        for (int i=0; i < HWC; i++) {
            buf[i] = static_cast<U8>(*p++ * 256.0);  /// * [0,1)=>[0,256)
        }
        fout.write((const char*)buf, HWC);
    }
    free(buf);
    return 0;
}

__HOST__ int
AIO::_tsave_npy(h_ostr &fout, Tensor &t) {
    // TODO:
    return 0;
}
#endif // T4_ENABLE_OBJ

