/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth Debug/Tracer implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "debug.h"
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ void
Debug::words(int rdx) {
    fout << std::setbase(10);
    for (int i=0, sz=0; i<_didx; i++) {
        fout << ' ';
        sz += to_s((IU)i) + 1;
        if (_trace || sz > 68) { fout << std::endl; sz = 0; } /// TODO: width configuable
    }
    if (!_trace) fout << std::endl;
}
///
/// recursively disassemble colon word
///
__HOST__ void
Debug::see(U8 *ip, int dp, int base) {
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
Debug::see(U16 w) {
    fout << "["; to_s(fout, w);
    if (_dict[w].colon) see(fout, &_pmem[_dict[w].pfa]);
    fout << "]" << std::endl;
}
///
/// dump data stack content
///
__HOST__ void
Debug::ss_dump(DU *ss, U16 n, int radix) {
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
Debug::mem_dump(U16 p0, U16 sz) {
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
__HOST__ void
Debug::dict_dump(int base) {                   ///< dump dictionary
    // TODO
}

__HOST__ void
Debug::mem_stat() {                            ///< display memory statistics
    // TODO
}
