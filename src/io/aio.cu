/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>       /// setbase, setprecision
#include "aio.h"

namespace t4::io {
///
///@name singleton and contructor
///@{
AIO *_io = NULL;         ///< singleton Async IO controller

__HOST__ AIO*
AIO::get_io(int *verbo) {
    if (!_io) _io = new AIO(verbo);
    return _io;
}
__HOST__ void AIO::free_io() { if (_io) delete _io; }
///@}
///@name object debugging method
///@{
__HOST__ void
AIO::setfmt(h_ostr &o, void *vp) {
    io::obuf_fmt *f = (io::obuf_fmt*)vp;
    DEBUG("  fmt: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
    o << std::setbase(f->base)
      << std::setw(f->width)
      << std::setprecision(f->prec ? f->prec : -1)
      << std::setfill((char)f->fill);
}

__HOST__ std::string
AIO::to_s(DU v, int base) {                        ///< display pure value
    static char buf[34];                           ///< static buffer
    DU t, f = modf(v, &t);                         ///< integral, fraction
    int i = 0;                                     
    if (ABS(f) > DU_EPS) {
        sprintf(buf, "%0.6g", v);
    }
    else {                                         ///< by-digit (Forth's <# #S #>)
        int dec = base==10;                        ///< C++ can do only base=8,10,16
        U32 n   = dec ? (U32)(ABS(v)) : (U32)(v);  ///< handle negative
        i = 33;  buf[i]='\0';                      /// * C++ can do only base=8,10,16
        do {                                       ///> digit-by-digit
            U8 d = (U8)MOD(n,base);  n /= base;
            buf[--i] = d > 9 ? (d-10)+'a' : d+'0';
        } while (n && i);
        if (dec && v < DU0) buf[--i]='-';
    }
    std::string s(&buf[i]);
    return s;
}

__HOST__ std::string
AIO::to_s(void *vp, U8 gt) {
    std::ostringstream ss;
    switch (gt) {
    case GT_INT:   ss << (*(S32*)vp);               break;
    case GT_U32:   ss << (*(U32*)vp);               break;
    case GT_FLOAT: ss << (*(DU*)vp);                break;
    case GT_STR:   ss << (char*)vp;                 break;
    case GT_OBJ:   ss << "ERROR: see sys#marshall"; break;
    case GT_FMT:   ss << "ERROR: see debug#print";  break;
    }
    DEBUG("  aio#print(fs, *v=0x%08x=%g, gt=%x)\n", DU2X(*(DU*)vp), *(DU*)vp, gt);
    
    return ss.str();
}

#if T4_DO_OBJ
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
#endif // T4_DO_OBJ

} // namespace t4::io
