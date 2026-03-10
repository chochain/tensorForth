/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *        Tensor & NN model persistence (i.e. serialization) methods
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>       /// setprecision
#include <fstream>
#include "aio.h"

#if T4_DO_OBJ

namespace t4::io {

__HOST__ int
AIO::tsave(Tensor &t, char *fname, U8 mode) {
    IO_DB("\nAIO::save tensor to '%s' {", fname);
    
    std::ios_base::openmode m =
        (mode & FAM_RW) ? std::ios_base::in : std::ios_base::out;
    if (mode & FAM_RAW) m |= std::ios_base::binary;
    
    std::fstream fs(fname, m);                    ///< open an output file
    if (!fs.is_open()) {
        ERROR(" failed to open for output\n");
        return 1;
    }
    if (mode & FAM_RAW) _tsave_raw(fs, t);        /// * write in raw format
    else                _tsave_txt(fs, t);        /// * write in text format
    
    fs.close();
    IO_DB("} => completed\n");
    return 0;
}
///
/// Tensor IO private methods
///
__HOST__ std::string
AIO::_vec(DU *vd, U32 W, U32 C) {
    std::ostringstream ss;
    auto num = [&ss, &C](DU *dx) {
        for (U32 k=0; k < C; k++) {
            ss << (k>0 ? "_" : " ") << *dx++;
        }
    };
    ss.flags(std::ios::showpos | std::ios::right | std::ios::fixed);   /// enforce +- sign
    ss.precision(_prec);                         ///< set precision
    ss << "{";
    U32 rw = (W <= _thres) ? W : (W < _edge ? W : _edge);
    for (U32 j=0; j < rw; j++) {                 ///< leading elements
        num(vd + j * C);
    }
    U32 x = W - rw;
    if (x > rw) ss << " ...";                    ///< colomn break
    
    for (U32 j=(x > rw ? x : rw); j < W; j++) {  ///< tailing elements
        num(vd + j * C);
    }
    ss << " }";
    return ss.str();
}

__HOST__ std::string
AIO::_mat(DU *td, U32 *shape) {
    auto range = [this](U32 v) { return (v < _edge) ? v : _edge; };
    const U32 H = shape[0], W = shape[1], C = shape[2]; ///< height, width, channels
    const U64 WC= (U64)W * C;
    const int rh= range(H), rw=range(W);                ///< h,w range for ...
    
    std::ostringstream ss;
    auto row = [this, &ss, &H, &W, &C](U32 y, DU *d) {
        ss << _vec(d, W, C) << (y == H ? "" : "\n\t");
    };
    
    DU *d = td;
    for (U32 y=0, y1=1; y<rh; y++, y1++, d+=WC) {   ///< leading rows
        row(y1, d);
    }

    U32 ym = (H <= _thres) ? rh : H - rh;
    if (ym > rh) ss << "...\n\t";                   ///< row break
    else ym = rh;
    
    d = td + ym * WC;
    for (U32 y=ym, y1=y+1; y<H; y++, y1++, d+=WC) { ///< trailing rows
        row(y1, d);
    }
    return ss.str();
}
__HOST__ std::string
AIO::_tensor(Tensor &t) {
    DU *td = t.data;                                    /// * short hand
    DEBUG("  aio#print_tensor T=%p data=%p\n", &t, td);
    std::ostringstream ss;

    switch (t.rank) {
    case 1: {
        ss << "vector" << shape(t) << " = ";
        ss << _vec(td, t.numel, 1);
    } break;
    case 2: {
        ss << "matrix" << shape(t) << " = {\n\t";
        ss << _mat(td, t.shape);
        ss << " }";
    } break;
    case 4: {
        int N = t.N();
        ss << "tensor" << shape(t) << " = { {\n\t";
        for (int n = 0; n < N; n++, td += t.HWC()) {
            ss << _mat(td, t.shape);
            ss << ((n+1) < N ? " } {\n\t" : "");
        }
        ss << " } }";
    } break;
    case 5: {
        ss << "tensor[" << t.iparm << "]" << shape(t) << " = {...}";
    } break;        
    default: ss << "tensor rank=" << t.rank << " not supported";
    }
    ss << '\n';
    return ss.str();
}
///
/// Tensor & NN model persistence (i.e. serialization) methods
///
__HOST__ int
AIO::_tsave_txt(h_ostr &fs, Tensor &t) {
    int tmp = _thres;
    _thres  = 1024;                                     /// * allow 1K*1K cells
    fs << _tensor(t);              
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
    /// TODO:
    return 0;
}

} // namespace t4::io

#endif // T4_DO_OBJ

