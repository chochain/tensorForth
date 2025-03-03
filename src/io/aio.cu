/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "aio.h"
///
///@name singleton and contructor
///@{
AIO *_io = NULL;         ///< singleton Async IO controller

__HOST__ AIO*
AIO::get_io() {
    if (!_io) _io = new AIO();
    return _io;
}
__HOST__ void AIO::free_io() { if (_io) delete _io; }
///@}
///@name object debugging method
///@{
__HOST__ void
AIO::print(h_ostr &fs, void *v, U8 gt) {
    switch (gt) {
    case GT_INT:   fs << (*(S32*)v); break;
    case GT_U32:   fs << (*(U32*)v); break;
    case GT_FLOAT: fs << (*(DU*)v);  break;
    case GT_STR:   fs << (char*)v;   break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        DEBUG("FMT: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        fs << std::setbase(f->base)
           << std::setw(f->width)
           << std::setprecision(f->prec ? f->prec : -1)
           << std::setfill((char)f->fill);
    } break;
    }
}
#if T4_ENABLE_OBJ
__HOST__ void
AIO::print(h_ostr &fs, T4Base &t) {
    switch (t.ttype) {
    case T4_TENSOR:
    case T4_DATASET: _print_tensor(fs, (Tensor&)t); break;
#if T4_ENABLE_NN        
    case T4_MODEL:   _print_model(fs, (Model&)t);   break;
#endif // T4_ENABLE_NN        
    case T4_XXX:     /* reserved */                   break;
    }
}
///@}
#endif // T4_ENABLE_OBJ    



