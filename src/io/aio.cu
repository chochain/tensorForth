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

__HOST__ void
AIO::to_s(DU v, int rdx) {
    fout << std::setbase(rdx) << v << std::setbase(_radix);
}
///
/// Object debugging methods
///
#if T4_ENABLE_OBJ
__HOST__ void
AIO::to_s(T4Base &t, bool is_view, int rdx) {
    _show_obj(t, is_view, rdx);
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

__HOST__ int
AIO::_show_obj(T4Base &t, bool is_view, int rdx) {
    static const char tn[2][4] = {                   ///< sync with t4_obj
        { 'T', 'N', 'D', 'X' }, { 't', 'n', 'd', 'x' }
    };
    auto t2 = [this](Tensor &t) { fout << t.H() << ',' << t.W() << ']'; };
    auto t4 = [this](Tensor &t) {
        fout << t.N() << ',' << t.H() << ',' << t.W() << ',' << t.C() << ']';
    };
    fout << std::setbase(rdx) << tn[is_view][t.ttype];
    switch(t.rank) {
    case 0: fout << "["  << (t.numel - 1) << "]"; break; // network model
    case 1: fout << "1[" << t.numel << "]";       break;
    case 2: fout << "2["; t2((Tensor&)t);         break;
    case 4: fout << "4["; t4((Tensor&)t);         break;
    case 5: fout << "5[" << t.parm << "]["; t4((Tensor&)t); break;
    }
    fout << std::setbase(_radix);
    return 1;
}
#endif // T4_ENABLE_OBJ    



