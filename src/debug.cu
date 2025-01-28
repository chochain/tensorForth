/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth Debug/Tracer implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>
#include "debug.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
__HOST__ int
Debug::to_s(IU w) {                     ///< dictionary name size
    h_ostr &fout = io->fout;
    /*
     * TODO: not sure why copying 32 byte does not work?
     * char name[36];
     * cudaMemcpy(name, _dict[w].name, 32, D2H);
     */
    Code &code = mu->_dict[w];
    if (io->trace) {
        fout << (code.immd ? "*" : " ")
             << "[" << std::setw(3) << w << "]"
             << (code.colon ? (FPTR)&mu->_pmem[code.nfa] : code.xt)
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
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ void
Debug::words(int rdx) {
    h_ostr &fout = io->fout;
    fout << std::setbase(10);
    for (int i=0, sz=0; i<mu->_didx; i++) {
        fout << ' ';
        sz += to_s((IU)i) + 1;
        if (io->trace || sz > 68) { fout << std::endl; sz = 0; } /// TODO: width configuable
    }
    if (!io->trace) fout << std::setbase(rdx) << std::endl;
}
///
/// recursively disassemble colon word
///
__HOST__ void
Debug::see(U8 *ip, int dp, int rdx) {
    h_ostr &fout = io->fout;
    while (*(IU*)ip) {                                              /// * loop until EXIT
        fout << std::endl; for (int n=dp; n>0; n--) fout << "  ";   /// * indentation by level
        fout << "[" << std::setw(4) << (IU)(ip - mu->_pmem) << ":";
        IU w = *(IU*)ip;                                            /// * fetch word index
        to_s(w);                                                    /// * display word name
        if (mu->_dict[w].colon && dp < 2) {                         /// * check if is a colon word
            see(&mu->_pmem[mu->_dict[w].pfa], dp+1);                /// * go one level deeper
        }
        ip += sizeof(IU);                                           /// * advance instruction pointer
        switch (w) {
        case DOVAR: case DOLIT: {                                   /// * fetch literal
            DU v = *(DU*)ip;  ip += sizeof(DU);
            fout << "= ";
            io->show(v);
            if (IS_OBJ(v)) io->show(v);                             /// * handle object
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
Debug::see(IU w, int rdx) {
    h_ostr &fout = io->fout;
    fout << "["; to_s(w);
    if (mu->_dict[w].colon) {
        see(&mu->_pmem[mu->_dict[w].pfa], 0);
    }
    fout << "]" << std::endl;
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
Debug::mem_dump(U32 p0, IU sz, int rdx) {
    static const char *i2h = "0123456789abcdef";
    h_ostr &fout = io->fout;
    char buf[80];
    for (U16 i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (U16 j=0; j<16; j++) {
            //U8 c = *(((U8*)&_dict[0])+i+j) & 0x7f;               // to dump _dict
            U8 c = mu->_pmem[i+j];
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
Debug::dict_dump(int rdx) {                    ///< dump dictionary
    // TODO
}

__HOST__ void
Debug::mem_stat() {                            ///< display memory statistics
    // TODO
}
///
/// dump data stack content
///
__HOST__ void
Debug::ss_dump(IU vid, U16 n, int rdx) {
    h_ostr &fout = io->fout;
    DU *ss = mu->vmss(vid);
    if (io->trace) fout << vid << "}";
    fout << std::setprecision(-1)
         << std::setbase(rdx) << " <";
    for (U16 i=0; i<n; i++) {
        io->show(ss[i]);            /// * show stack elements
        fout << " ";
    }
    io->show(ss[T4_SS_SZ-1]);       /// * show top
    fout << "> ok" << std::endl;
}
