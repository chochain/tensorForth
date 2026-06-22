/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *        Tensor & NN model persistence (i.e. serialization) methods
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <vector>
#include "aio.h"
#include "mu/tensor.h"

#if T4_DO_OBJ

namespace t4::io {

__HOST__ std::string
AIO::to_s(T4Base &t, bool view) {
    static const char tn[2][4] = {                ///< sync with t4_obj
        { 'T', 'N', 'D', 'X' }, { 't', 'n', 'd', 'x' }
    };
    std::ostringstream ss;
    
    ss << tn[view][t.ttype];
    switch(t.rank) {
    case 0:            break;                     ///< network model
    case 1: ss << '1'; break;
    case 2: ss << '2'; break;
    case 3: ss << '3'; break;
    case 4: ss << '4'; break;
    case 5: ss << "5[" << t.iparm << "]"; break;
    }
    ss << shape(t);
    
    return ss.str();
}

__HOST__ std::string
AIO::shape(T4Base &b) {
    Tensor &t = (Tensor&)b;
    std::ostringstream ss;

    ss << '[';
    switch (t.rank) {
    case 0: ss << (t.numel - 1);         break;   /// network model
    case 1: ss << t.numel;               break;
    case 2: ss << t.H() << ',' << t.W(); break;
    case 3: ss << "na";                  break;
    case 4:
    case 5: ss << t.N() << ',' << t.H() << ','
               << t.W() << ',' << t.C(); break;
    }
    ss << ']';
    
#if MM_DEBUG
    if (t.rank==2 || t.rank==5) ss << t.numel;
#endif // MM_DEBUG
    
    return ss.str();
}

__HOST__ std::string
AIO::marshall(T4Base &t) {
    DEBUG("  aio#print(fs, t4base=%p)\n", &t);
    switch (t.ttype) {
    case T4_TENSOR:
    case T4_DATASET: return _tensor((Tensor&)t);
#if T4_DO_NN
    case T4_MODEL:   return _model((Model&)t);
#endif // T4_DO_NN        
    }
    return std::string("");
}
///@}

__HOST__ int
AIO::tsave(Tensor &t, char *fname, U8 mode) {
    IO_DB("AIO::save tensor to '%s' {", fname);
    
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

#define  STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__HOST__ int
AIO::t2png(Tensor &t, char *tag, int n_per_row) {
    const int N     = t.N(), H = t.H(), W = t.W(), C = t.C();
    const int WT    = n_per_row * W;
    const int HT    = (N + n_per_row - 1) / n_per_row;
#if __CUDACC__    
    const DU  mean  = t.avg(), scale = 64.0f / t.std();   ///< 2 std = 95%
#else  // !__CUDACC__
    const DU  mean  = 128.0f, scale = 128.0f;             ///< [0,256)
#endif // __CUDACC__    

    auto tile = [&](U8 *px, DU *v, int idx) {
        int ht = idx / n_per_row, wt = idx % n_per_row;
        U8 *p = &px[(ht * H * WT + wt * W) * 3];
        for (int y = 0; y < H; y++) {
            for (int x = 0, c = 0; x < W; x++, c=0) {
                while (c < 3) {                  /// RGB
                    DU vx = (*v - mean) * scale;
                    *p++ = (U8)fminf(255.0f, fmaxf(vx, 0.0f));
                    if (c++ < C) v++;            /// advance if more than 1 channel
                }
            }
            p += (WT - W) * 3;                   /// skip to next row in tile
        }
    };

    U8 px[(HT * H) * WT * 3] = {};               ///< zero-init, so unfilled are black
#if __CUDACC__    
    std::vector<DU> h(t.numel);                  ///< host block (on heap space)
    D2H(h.data(), t.data, sizeof(DU) * t.numel); ///< copy from device to host
#else  // !__CUDACC__
    DU *h = t.data;
#endif // __CUDACC__    
    for (int n = 0; n < N; n++) {
        DU *hx = &h[n * H * W * C];
        tile(px, hx, n);
    }
    /// stride must be WT*3 (full tiled row), not W*3
    if (!stbi_write_png(tag, WT, H * HT, 3, px, WT * 3)) {
        ERROR("%s write failed\n", tag); return -1;
    }
    return 0;
}
///
/// Tensor IO private methods
///
__HOST__ std::string
AIO::_vec(DU *vd, U32 W, U32 C) {
    std::ostringstream ss;
    auto num = [&ss, C](DU *dx) {
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
    auto range = [this](U32 v) {
        return (v < _edge) ? v : _edge;
    };
    const U32 H = shape[0], W = shape[1], C = shape[2]; ///< height, width, channels
    const U64 WC= (U64)W * C;
    const U32 rh= range(H);                             ///< h range for ...
    
    std::ostringstream ss;
    auto row = [this, &ss, H, W, C](U32 y, DU *d) {
        ss << _vec(d, W, C) << (y == H ? "" : "\n\t");
    };
    
    DU *d = td;
    for (U32 y=0, y1=1; y < rh; y++, y1++, d+=WC) {     ///< leading rows
        row(y1, d);
    }

    U32 ym = (H <= _thres) ? rh : H - rh;
    if (ym > rh) ss << "...\n\t";                       ///< row break
    else ym = rh;
    
    d = td + ym * WC;
    for (U32 y=ym, y1=y+1; y<H; y++, y1++, d+=WC) {     ///< trailing rows
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
AIO::_tsave_txt(ostr &fs, Tensor &t) {
    int tmp = _thres;
    _thres  = 1024;                                     /// * allow 1K*1K cells
    fs << _tensor(t);
    _thres  = tmp;
    return 0;
}

__HOST__ int
AIO::_tsave_raw(ostr &fs, Tensor &t) {
    const char hdr[2] = { 'T', '4' };
    const int N = t.N(), HWC = t.HWC();
    std::vector<U8> buf(HWC);                           ///< buffer for one slice

    fs.write(hdr, sizeof(hdr));
    fs.write((const char*)t.shape, sizeof(t.shape));
    for (int n=0; n < N; n++) {
        DU *p = &t.data[n * HWC];                       ///< slice-by-slice
        for (int i=0; i < HWC; i++) {
            buf[i] = static_cast<U8>(*p++ * 256.0);     /// * [0,1)=>[0,256)
        }
        fs.write((const char*)buf.data(), HWC);
    }
    return 0;
}

__HOST__ int
AIO::_tsave_npy(ostr &fs, Tensor &t) {
    /// TODO:
    return 0;
}

} // namespace t4::io

#endif // T4_DO_OBJ

