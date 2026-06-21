/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>       /// setbase, setprecision
#include "ten4_types.h"
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
AIO::setfmt(ostr &o, void *vp) {
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
    DU t, f = modf(v, &t);                         ///< split v into integral, fraction
    int dec = base==10;                            ///< decimal output
    int i = 0;                                     
    if (dec && abs(f) > DU_EPS) {                  ///< number with fractions
        sprintf(buf, "%0.6g", v);
    }
    else {                                         ///< by-digit (Forth's <# #S #>)
        U32 n = dec ? (U32)(abs(v)) : (U32)(v);    ///< handle negative
        i = 33;  buf[i]='\0';                      /// * C++ can do only base=8,10,16
        do {                                       ///> digit-by-digit
            U8 d = (U8)(n % base);  n /= base;
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

} // namespace t4::io
