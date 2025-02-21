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
AIO::show(DU v, int rdx) {
#if T4_ENABLE_OBJ    
    if (IS_OBJ(v)) { _show_obj(T4Base::du2obj(v), IS_VIEW(v)); return; }
#endif
    if (rdx != 10) fout << static_cast<int>(v);
    else           fout << v;
}

__HOST__ void
AIO::print(DU v) {
#if T4_ENABLE_OBJ
    T4Base &o = T4Base::du2obj(v);
    switch (o.ttype) {
    case T4_TENSOR:
    case T4_DATASET: _print_tensor((Tensor&)o); break;
#if T4_ENABLE_NN        
    case T4_MODEL:   _print_model((Model&)o);   break;
#endif // T4_ENABLE_NN        
    case T4_XXX:     /* reserved */             break;
    }
    return;
#endif // T4_ENABLE_OBJ
    fout << v;
}
///
/// Object debugging methods
///
#if T4_ENABLE_OBJ
__HOST__ int
AIO::_show_obj(T4Base &t, bool view) {
    static const char tn[2][4] = {                   ///< sync with t4_obj
        { 'T', 'N', 'D', 'X' }, { 't', 'n', 'd', 'x' }
    };
    auto t2 = [&fout](Tensor &t) { fout << t.H() << ',' << t.W() << ']'; };
    auto t4 = [&fout](Tensor &t) {
        fout << t.N() << ',' << t.H() << ',' << t.W() << ',' << t.C() << ']';
    };
    fout << tn[view][t.ttype];
    switch(t.rank) {
    case 0: fout << "["  << (t.numel - 1) << "]"; break; // network model
    case 1: fout << "1[" << t.numel << "]";       break;
    case 2: fout << "2["; t2((Tensor&)t);         break;
    case 4: fout << "4["; t4((Tensor&)t);         break;
    case 5: fout << "5[" << t.parm << "]["; t4((Tensor&)t); break;
    }
    return 1;
}
#endif // T4_ENABLE_OBJ    



