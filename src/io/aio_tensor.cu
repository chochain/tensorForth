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
using namespace std;
///
/// Tensor & NN model persistence (i.e. serialization) methods
///
#include <fstream>
#include "mmu/tensor.h"

__HOST__ int
AIO::tsave(Tensor &t, char *fname, U8 mode) {
    IO_DB("\nAIO::save tensor to '%s' =>", fname);
    
    ios_base::openmode m = (mode & FAM_RW) ? ios_base::in : ios_base::out;
    if (mode & FAM_RAW) m |= ios_base::binary;
    
    fstream fs(fname, m);                         ///< open an output file
    if (!fs.is_open()) {
        ERROR(" failed to open for output\n");
        return 1;
    }
    if (mode & FAM_RAW) _tsave_raw(fs, t);        /// * write in raw format
    else                _tsave_txt(fs, t);        /// * write in text format
    
    fs.close();
    IO_DB(" completed\n");
    return 0;
}
///
/// Tensor IO private methods
///
__HOST__ void
AIO::_print_vec(h_ostr &fs, DU *vd, U32 W, U32 C) {
    U32 rw = (W <= _thres) ? W : (W < _edge ? W : _edge);
    fs << setprecision(_prec) << "{";             /// set precision
    for (U32 j=0; j < rw; j++) {
        DU *dx = vd + j * C;
        for (U32 k=0; k < C; k++) {
            fs << (k>0 ? "_" : " ") << *dx++;
        }
    }
    U32 x = W - rw;
    if (x > rw) fs << " ...";
    for (U32 j=(x > rw ? x : rw); j < W; j++) {
        DU *dx = vd + j * C;
        for (U32 k=0; k < C; k++) {
            fs << (k>0 ? "_" : " ") << *dx++;
        }
    }
    fs << " }";
}
__HOST__ void
AIO::_print_mat(h_ostr &fs, DU *td, U32 *shape) {
    auto range = [this](U32 v) { return (v < _edge) ? v : _edge; };
    const U32 H = shape[0], W = shape[1], C = shape[2]; ///< height, width, channels
    const int rh= range(H), rw=range(W);                ///< h,w range for ...
    DU *d = td;
    
    fs.flags(ios::showpos | ios::right | ios::fixed);   /// enforce +- sign
    for (U32 y=0, y1=1; y<rh; y++, y1++, d+=(W * C)) {
        _print_vec(fs, d, W, C);
        fs << (y1==H ? "" : "\n\t");
    }

    U32 ym = (H <= _thres) ? rh : H - rh;
    if (ym > rh) fs << "...\n\t";
    else ym = rh;
    
    d = td + ym * W * C;
    for (U32 y=ym, y1=y+1; y<H; y++, y1++, d+=(W * C)) {
        _print_vec(fs, d, W, C);
        fs << (y1==H ? "" : "\n\t");
    }
}
__HOST__ void
AIO::_print_tensor(h_ostr &fs, Tensor &t) {
    DU *td = t.data;                                    /// * short hand
    IO_DB("aio#print_tensor::T=%p data=%p\n", &t, td);

    ios::fmtflags fmt0 = fs.flags();
    fs << setprecision(-1);                             /// * standard format
    switch (t.rank) {
    case 1: {
        fs << "vector[" << t.numel << "] = ";
        _print_vec(fs, td, t.numel, 1);
    } break;
    case 2: {
        fs << "matrix[" << t.H() << "," << t.W() << "] = {\n\t";
        _print_mat(fs, td, t.shape);
        fs << " }";
    } break;
    case 4: {
        int N = t.N();
        fs << "tensor["
           << N << "," << t.H() << "," << t.W() << "," << t.C()
           << "] = { {\n\t";
        for (int n = 0; n < N; n++, td += t.HWC()) {
            _print_mat(fs, td, t.shape);
            fs << ((n+1) < N ? " } {\n\t" : "");
        }
        fs << " } }";
    } break;
    case 5: {
        fs << "tensor[" << t.parm << "]["
           << t.N() << "," << t.H() << "," << t.W() << "," << t.C()
           << "] = {...}";
    } break;        
    default: fs << "tensor rank=" << t.rank << " not supported";
    }
    fs << "\n";
    fs.flags(fmt0);
}
///
/// Tensor & NN model persistence (i.e. serialization) methods
///
__HOST__ int
AIO::_tsave_txt(h_ostr &fs, Tensor &t) {
    int tmp = _thres;
    _thres  = 1024;                                     /// * allow 1K*1K cells
    _print_tensor(fs, t);              
    _thres  = tmp;
    return 0;
}

__HOST__ int
AIO::_tsave_raw(h_ostr &fs, Tensor &t) {
    const char hdr[2] = { 'T', '4' };
    const int N = t.N(), HWC = t.HWC();
    U8 *buf = (U8*)malloc(HWC);                         ///< buffer for one slice

    fs.write(hdr, sizeof(hdr));
    fs.write((const char*)t.shape, sizeof(t.shape));
    for (int n=0; n < N; n++) {
        DU *p = &t.data[n * HWC];                       ///< slice-by-slice
        for (int i=0; i < HWC; i++) {
            buf[i] = static_cast<U8>(*p++ * 256.0);     /// * [0,1)=>[0,256)
        }
        fs.write((const char*)buf, HWC);
    }
    free(buf);
    return 0;
}

__HOST__ int
AIO::_tsave_npy(h_ostr &fs, Tensor &t) {
    // TODO:
    return 0;
}
#endif // T4_ENABLE_OBJ

