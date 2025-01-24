/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth System interface implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>                     /// setw, setbase
#include "sys.h"
///
/// random number generator setup
/// Note: kept here because curandStates stays in CUDA memory
///
__KERN__ void
k_rand_init(curandState *st, U64 seed) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, x, 0, &st[x]);
}

__KERN__ void
k_rand(DU *mat, int sz, DU bias, DU scale, curandState *st, t4_rand_opt ntype) {
    int tx = threadIdx.x;             ///< thread idx
    int n  = (sz / blockDim.x) + 1;   ///< loop counter
    
    curandState s = st[tx];           /// * cache state into local register
    for (int i=0, x=tx; i<n; i++, x+=blockDim.x) {  /// * scroll through pages
        if (x < sz) {
            mat[x]= scale * (
                bias + (ntype==NORMAL ? curand_normal(&s) : curand_uniform(&s))
                );
        }
    }
    st[tx] = s;                      /// * copy state back to global memory
}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
System::System() {
    MM_ALLOC(&_seed, sizeof(curandState) * T4_RAND_SZ);
    
    k_rand_init<<<1, T4_RAND_SZ>>>(_seed, time(NULL));  /// serialized randomizer
    GPU_CHK();
}

System::~System() {
    MM_FREE(_seed);
}

__GPU__ DU
System::rand(DU d, t4_rand_opt ntype) {
    if (!IS_OBJ(d)) return d * curand_uniform(&_seed[0]);
    random((Tensor&)du2obj(d), ntype);
    return d;
}
///
/// Object debugging methods
///
__HOST__ int
System::to_s(std::ostream &fout, T4Base &t, bool view) {
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

__HOST__ __INLINE__ int
System::to_s(std::ostream &fout, DU s) {
    return to_s(fout, du2obj(s), IS_VIEW(s));
}

#endif // T4_ENABLE_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// Debugging methods
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ int
System::to_s(std::ostream &fout, IU w) {
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
/// display dictionary word list
///
__HOST__ void
System::words(std::ostream &fout) {
    fout << std::setbase(10);
    for (int i=0, sz=0; i<_didx; i++) {
        fout << ' ';
        sz += to_s(fout, (IU)i) + 1;
        if (_trace || sz > 68) { fout << std::endl; sz = 0; } /// TODO: width configuable
    }
    if (!_trace) fout << std::endl;
}
///
/// recursively disassemble colon word
///
__HOST__ void
System::see(std::ostream &fout, U8 *ip, int dp) {
    while (*(IU*)ip) {                                              /// * loop until EXIT
        fout << std::endl; for (int n=dp; n>0; n--) fout << "  ";   /// * indentation by level
        fout << "[" << std::setw(4) << (IU)(ip - _pmem) << ":";
        IU w = *(IU*)ip;                                            /// * fetch word index
        to_s(fout, w);                                              /// * display word name
        if (_dict[w].colon && dp < 2) {                             /// * check if is a colon word
            see(fout, &_pmem[_dict[w].pfa], dp+1);                  /// * go one level deeper
        }
        ip += sizeof(IU);                                           /// * advance instruction pointer
        switch (w) {
        case DOVAR: case DOLIT: {                                   /// * fetch literal
            DU v = *(DU*)ip;  ip += sizeof(DU);
            fout << "= ";
            if (IS_OBJ(v)) to_s(fout, v);                           /// * handle object
            else fout << v;                                         /// * display the literal
        } break;
        case DOSTR: case DOTSTR: {
            char *s = (char*)ip;
            int  sz = strlen(s)+1;
            ip += ALIGN2(sz);                                       /// fetch string
            fout << "= \"" << s << "\"";
        } break;
        case BRAN: case ZBRAN: case DONEXT:
            fout << "= " << *(IU*)ip; ip += sizeof(IU); break;      /// fetch jump target
        }
        fout << " ] ";
    }
}
__HOST__ void
System::see(std::ostream &fout, U16 w) {
    fout << "["; to_s(fout, w);
    if (_dict[w].colon) see(fout, &_pmem[_dict[w].pfa]);
    fout << "]" << std::endl;
}
///
/// dump data stack content
///
__HOST__ void
System::ss_dump(std::ostream &fout, U16 vid, U16 n, int radix) {
    bool rx = radix != 10;
    auto show = [this, &fout, rx](DU s) {
        if (IS_OBJ(s)) to_s(fout, s);
        else if (rx)   fout << static_cast<int>(s);
        else           fout << s;
    };
    DU *ss = &_vmss[vid * T4_SS_SZ];
    if (_trace) fout << vid << "}";
    fout << std::setprecision(-1) << " <";
    if (rx) fout << std::setbase(radix);
    for (U16 i=0; i<n; i++) {
        show(ss[i]);                 /// * show stack elements
        fout << " ";
    }
    show(ss[T4_SS_SZ-1]);            /// * show top
    fout << "> ok" << std::endl;
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
System::mem_dump(std::ostream &fout, U16 p0, U16 sz) {
    static const char *i2h = "0123456789abcdef";
    char buf[80];
    for (U16 i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (U16 j=0; j<16; j++) {
            //U8 c = *(((U8*)&_dict[0])+i+j) & 0x7f;               // to dump _dict
            U8 c = _pmem[i+j];
            C2H(c);                                                // "%02x "
            c &= 0x7f;                                             // mask off high bit
            buf[x++] = ' ';
            if (j%4==3) buf[x++] = ' ';
            buf[59+j]= (c==0x7f||c<0x20) ? '.' : c;                // %c
        }
        buf[75] = '\0';
        fout << buf;
    }
    fout << std::endl;
}

