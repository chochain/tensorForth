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
AIO::get_io(h_istr &i, h_ostr &o, int verbo) {
    if (!_io) _io = new AIO(i, o, verbo);
    return _io;
}
__HOST__ AIO *AIO::get_io()  { return _io; }
__HOST__ void AIO::free_io() { if (_io) delete _io; }
///@}
///@name simple value and object debugging method
///@{
__HOST__ void
AIO::show(DU v, int base) {               ///< display value by ss_dump
    static char buf[34];                  ///< static buffer
    auto pp = [](DU v, int b) {           ///< display v by radix
        DU t, f = modf(v, &t);            ///< integral, fraction
        if (ABS(f) > DU_EPS) {
            sprintf(buf, "%0.6g", v);
            return buf;
        }
        int i   = 33;  buf[i]='\0';       /// * C++ can do only base=8,10,16
        int dec = b==10;
        U32 n   = dec ? (U32)(ABS(v)) : (U32)(v);  ///< handle negative
        do {                              ///> digit-by-digit
            U8 d = (U8)MOD(n,b);  n /= b;
            buf[--i] = d > 9 ? (d-10)+'a' : d+'0';
        } while (n && i);
        if (dec && v < DU0) buf[--i]='-';
        return &buf[i];
    };
    fout << std::setbase(base)
         << pp(v, base) << ' '
         << std::setbase(_radix);
}

#if T4_ENABLE_OBJ
__HOST__ void                                  ///< display value by ss_dump
AIO::show(T4Base &t, bool is_view, int base) {
    _show_obj(t, is_view, base);
}

__HOST__ void
AIO::print(T4Base &t) {
    switch (t.ttype) {
    case T4_TENSOR:
    case T4_DATASET: _print_tensor(fout, (Tensor&)t); break;
#if T4_ENABLE_NN        
    case T4_MODEL:   _print_model(fout, (Model&)t);   break;
#endif // T4_ENABLE_NN        
    case T4_XXX:     /* reserved */                   break;
    }
}
///@}
///==========================================================================
///@name private methods
///@{
__HOST__ int
AIO::_show_obj(T4Base &t, bool is_view, int base) {
    static const char tn[2][4] = {                   ///< sync with t4_obj
        { 'T', 'N', 'D', 'X' }, { 't', 'n', 'd', 'x' }
    };
    auto t2 = [this](Tensor &t) { fout << t.H() << ',' << t.W() << ']'; };
    auto t4 = [this](Tensor &t) {
        fout << t.N() << ',' << t.H() << ',' << t.W() << ',' << t.C() << ']';
    };
    fout << std::setbase(base) << tn[is_view][t.ttype];
    switch(t.rank) {
    case 0: fout << "["  << (t.numel - 1) << "]"; break; // network model
    case 1: fout << "1[" << t.numel << "]";       break;
    case 2: fout << "2["; t2((Tensor&)t);         break;
    case 4: fout << "4["; t4((Tensor&)t);         break;
    case 5: fout << "5[" << t.parm << "]["; t4((Tensor&)t); break;
    }
    fout << ' ' << std::setbase(_radix);
    return 1;
}
///@}
#endif // T4_ENABLE_OBJ    



