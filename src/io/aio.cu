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
AIO::get_io(int *verbo) {
    if (!_io) _io = new AIO(verbo);
    return _io;
}
__HOST__ void AIO::free_io() { if (_io) delete _io; }
///@}
///@name object debugging method
///@{
__HOST__ void
AIO::to_s(h_ostr &fs, DU v, int base) {            ///< display value by ss_dump
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
    fs << &buf[i];
}
__HOST__ void
AIO::print(h_ostr &fs, void *v, U8 gt) {
    switch (gt) {
    case GT_INT:   fs << (*(S32*)v); break;
    case GT_U32:   fs << (*(U32*)v); break;
    case GT_FLOAT: fs << (*(DU*)v);  break;
    case GT_STR:   fs << (char*)v;   break;
    case GT_FMT:   {
        obuf_fmt *f = (obuf_fmt*)v;
        DEBUG("  fmt: b=%d, w=%d, p=%d, f='%c'\n", f->base, f->width, f->prec, f->fill);
        fs << std::setbase(f->base)
           << std::setw(f->width)
           << std::setprecision(f->prec ? f->prec : -1)
           << std::setfill((char)f->fill);
    } break;
    }
    DU n = *(DU*)v;
    DEBUG("  aio#print(fs, *v=0x%08x=%g, gt=%x)\n", DU2X(n), n, gt);
}
#if T4_DO_OBJ
///@}
///@name show simple value and object token for ss_dump
///@{
__HOST__ void
AIO::to_s(h_ostr &fs, T4Base &t, bool view) {
    static const char tn[2][4] = {                ///< sync with t4_obj
        { 'T', 'N', 'D', 'X' }, { 't', 'n', 'd', 'x' }
    };
    auto t2 = [this, &fs](Tensor &t) { fs << t.H() << ',' << t.W() << ']'; };
    auto t4 = [this, &fs](Tensor &t) {
        fs << t.N() << ',' << t.H() << ',' << t.W() << ',' << t.C() << ']';
    };
    fs << tn[view][t.ttype];
    switch(t.rank) {
    case 0: fs << "["  << (t.numel - 1) << ']';           break;  // network model
    case 1: fs << "1[" << t.numel << ']';                 break;
    case 2: fs << "2["; t2((Tensor&)t);                   break;
    case 3: fs << "3[na]";                                break;
    case 4: fs << "4["; t4((Tensor&)t);                   break;
    case 5: fs << "5[" << t.parm << "]["; t4((Tensor&)t); break;
    }
}
__HOST__ void
AIO::print(h_ostr &fs, T4Base &t) {
    switch (t.ttype) {
    case T4_TENSOR:
    case T4_DATASET: _print_tensor(fs, (Tensor&)t); break;
#if T4_DO_NN
    case T4_MODEL:   _print_model(fs, (Model&)t);   break;
#endif // T4_DO_NN        
    case T4_XXX:     /* reserved */                 break;
    }
    DEBUG("  aio#print(fs, t4base=%p)\n", &t);
}
///@}
#endif // T4_DO_OBJ    
