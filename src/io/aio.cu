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
///@name object debugging method
///@{
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
__HOST__ int
AIO::hint(h_ostr &fs, T4Base &t, bool view, int base) {
    static const char tn[2][4] = {                   ///< sync with t4_obj
        { 'T', 'N', 'D', 'X' }, { 't', 'n', 'd', 'x' }
    };
    auto t2 = [&fs](Tensor &t) { fs << t.H() << ',' << t.W() << ']'; };
    auto t4 = [&fs](Tensor &t) {
        fs << t.N() << ',' << t.H() << ',' << t.W() << ',' << t.C() << ']';
    };
    fs << std::setbase(base) << tn[view][t.ttype];
    switch(t.rank) {
    case 0: fs << "["  << (t.numel - 1) << "]"; break; // network model
    case 1: fs << "1[" << t.numel << "]";       break;
    case 2: fs << "2["; t2((Tensor&)t);         break;
    case 4: fs << "4["; t4((Tensor&)t);         break;
    case 5: fs << "5[" << t.parm << "]["; t4((Tensor&)t); break;
    }
    fs << ' ' << std::setbase(_radix);
    return 1;
}
///@}
#endif // T4_ENABLE_OBJ    



