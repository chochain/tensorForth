/** -*- c++ -*-
 * @file
 * @brief AIO class - async IO module implementation
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>        // printf
#include <iomanip>       // setbase, setprecision
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
__HOST__ __INLINE__ int
AIO::to_s(DU s) {
    return to_s(fout, du2obj(s), IS_VIEW(s));
}
__HOST__ int
AIO::to_s(IU w) {
    /*
     * TODO: not sure why copying 32 byte does not work?
     * char name[36];
     * cudaMemcpy(name, _dict[w].name, 32, D2H);
     */
    Code &code = _dict[w];
    if (_trace) {
        fout << (code.immd ? "*" : " ")
             << "[" << std::setw(3) << w << "]"
             << (code.colon ? (FPTR)&_pmem[code.nfa] : code.xt)
             << (code.colon ? ':': '=');
    }
    U8 c, i=0;
    cudaMemcpy(&c, code.name, 1, D2H);
    fout << " ";
    while (c) {
        fout << c;
        cudaMemcpy(&c, code.name+(++i), 1, D2H);
    }
    return (int)i;
}
///
/// Object debugging methods
///
__HOST__ int
AIO::to_s(T4Base &t, bool view) {
#if T4_ENABLE_OBJ
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
#endif // T4_ENABLE_OBJ
    return 1;
}
