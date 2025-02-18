/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth Debug/Tracer implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>
#include "debug.h"
///
///@name memory macros to reduce verbosity
///@{
#define MEM(a)   ((U8*)&mu->_pmem[a])         /** memory pointer by offset           */
#define DIDX     (mu->_didx)                  /** number of dictionary entries       */
#define DICT(w)  (mu->_dict[w])               /** dictionary entry                   */
#define XT0      ((UFP)DICT(0).xt)            /** base of lambda functions (i.e. xt) */
///@}
///@name Primitive words to help printing
///@{
Code prim[] = {
    Code(";",    EXIT),  Code("next ", NEXT), Code("loop ", LOOP), Code("lit",   LIT),
    Code("var",  VAR),   Code("str",   STR),  Code("dotq",  DOTQ), Code("bran ", BRAN),
    Code("0bran",ZBRAN), Code("for  ", FOR),  Code("do",    DO),   Code("key",   KEY)
};
///@}
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
__HOST__ void
Debug::ss_dump(IU id, int n, int base) {
    static char buf[34];                  ///< static buffer
    auto rdx = [](DU v, int b) {          ///< display v by radix
        DU t, f = modf(v, &t);            ///< integral, fraction
        if (ABS(f) > DU_EPS) {
            sprintf(buf, "%0.6g", v);
            return buf;
        }
        int i = 33;  buf[i]='\0';         /// * C++ can do only base=8,10,16
        int dec = b==10;
        U32 n   = dec ? (U32)(ABS(v)) : (U32)(v);  ///< handle negative
        do {                              ///> digit-by-digit
            U8 d = (U8)MOD(n,b);  n /= b;
            buf[--i] = d > 9 ? (d-10)+'a' : d+'0';
        } while (n && i);
        if (dec && v < DU0) buf[--i]='-';
        return &buf[i];
    };
    h_ostr &fout = io->fout;
    DU *ss = mu->vmss(id);                ///< retrieve VM SS
    for (int i=0; i < n; i++) {
        fout << rdx(*ss++, base) << ' ';
    }
    /// TODO << TOS
    fout << "-> ok" << std::endl;
}
__HOST__ int
Debug::p2didx(Param *p) {
    UFP xt0 = XT0;
    IU  pfa = p->ioff;
    for (int i = DIDX - 1; i > 0; --i) {
        Code &c  = DICT(i);
        bool hit = p->udf
            ? (c.udf  && pfa==c.pfa)
            : (!c.udf && pfa==(IU)((UFP)c.xt - xt0));
        if (hit) return i;
    }
    return -1;                                     /// * not found
}
__HOST__ int
Debug::to_s(IU w, int base) {
    Param *p = (Param*)MEM(DICT(w).pfa);
    return to_s(p, 0, base);
}
__HOST__ int
Debug::to_s(Param *p, int nv, int base) {
    bool pm = p->op != MAX_OP;                     ///< is prim
    int  w  = pm ? p->op : p2didx(p);              ///< fetch word index by pfa
    if (w < 0) return -1;                          ///> loop guard
    
    h_ostr &fout = io->fout;
    Code   &code = DICT(w);
    
    fout << std::endl << "  ";                     /// * indent
    if (io->trace) {                               /// * header
        fout << std::hex
             << std::setfill('0') << "( "
              << std::setw(4) << ((U8*)p - MEM(0)) ///> addr
             << std::setfill(' ') << '['
             << std::setw(4) << w << "] )"         ///> word ref
             << std::setbase(base);
    }
    if (!pm) {                                     ///> built-in
        char name[256];                            ///< name buffer on host
        d2h_strcpy(name, code.name);               /// * copy string from device
        fout << name << "  ";
        return 0;
    }
    U8 *ip = (U8*)(p+1);                           ///< pointer to data
    switch (w) {
    case LIT:  io->show(*(DU*)ip);                  break;
    case STR:  fout << "s\" " << (char*)ip << '"';  break;
    case DOTQ: fout << ".\" " << (char*)ip << '"';  break;
    case VAR:
        for (int i=0; i < nv; i+=sizeof(DU)) {
            fout << *(DU*)(ip + i) << ' ';
        }
        /* no break */
    default: fout << prim[w].name; break;
    }
    switch (w) {
    case NEXT: case LOOP:
    case BRAN: case ZBRAN:                   ///> display jmp target
        fout << " \\ $" << std::hex
             << std::setfill('0') << std::setw(4) << p->ioff;
        break;
    default: fout << std::setfill(' ') << std::setw(-1);          ///> restore format
    }
    return
        w==EXIT ||                           /// * end of word
        (w==LIT && p->exit) ||               /// * constant
        (w==VAR && !p->ioff);                /// * variable
}
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ void
Debug::words(int base) {
    const int WIDTH = 60;
    h_ostr &fout = io->fout;
    fout << std::dec;
    char name[256];
    for (int i=1, sz=0; i < DIDX; i++) {
        d2h_strcpy(name, DICT(i).name);
        fout << "  " << name;
        sz += strlen(name) + 2;

        if (sz > WIDTH) {
            fout << ENDL; sz = 0;
        }
    }
    fout << std::setbase(base) << std::endl;
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
Debug::mem_dump(IU p0, int sz, int base) {
    const char i2h[] = "0123456789abcdef";
    h_ostr &fout = io->fout;
    char buf[80];
    fout << std::hex << std::setfill('0');
    for (IU i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (IU j=0; j<16; j++) {
            //U8 c = *(((U8*)&_dict[0])+i+j) & 0x7f;               // to dump _dict
            U8 c = *MEM(i+j);
            C2H(c);                                                // "%02x "
            c &= 0x7f;                                             // mask off high bit
            buf[x++] = ' ';
            if (j%4==3) buf[x++] = ' ';
            buf[59+j]= (c==0x7f||c<0x20) ? '.' : c;                // %c
        }
        buf[75] = '\0';
        fout << buf;
    }
    fout << std::setfill(' ') << std::setbase(base) << std::endl;
}

#define NFA(w) (DICT(w).pfa - ALIGN(strlen(DICT(w).name)))
__HOST__ void
Debug::see(IU w, int base) {
    h_ostr &fout = io->fout;
    Code   &c    = DICT(w);
    fout << ": " << c.name << ENDL;
    if (!c.udf) {
        fout << " ( built-ins ) ;" << std::endl;
        return;
    }
    auto nvar = [this](IU i0, IU ioff, U8 *ip) {       /// * calculate # of elements
        if (ioff) return MEM(ioff) - ip - sizeof(IU);  /// create...does>
        IU pfa0 = DICT(i0).pfa;
        IU nfa1 = (i0+1) < DIDX ? NFA(i0+1) : mu->_midx;
        return (nfa1 - pfa0 - sizeof(IU));             ///> variable, create ,
    };
    U8 *ip = MEM(c.pfa);                               ///< PFA pointer
    while (1) {
        Param *p = (Param*)ip;
        int   nv = p->op==VAR ? nvar(w, p->ioff, ip) : 0;  ///< VAR number of elements
        if (to_s(p, nv, base) != 0) break;                 ///< display Parameter
        ///
        /// advance ip to next Param
        ///
        ip += sizeof(IU);
        switch (p->op) {                     ///> extra bytes to skip
        case LIT: ip += sizeof(DU);             break;
        case VAR: ip = MEM(p->ioff);            break;  ///> create/does
        case STR: case DOTQ: ip += p->ioff;     break;
        }
    }
    fout << std::endl;
}
///====================================================================
///
///> System statistics - for heap, stack, external memory debugging
///
__HOST__ void
Debug::dict_dump(int base) {
    h_ostr &fout = io->fout;
    UFP xt0 = XT0;
    char name[256];
    fout << "Built-in Dictionary: _XT0="
         << std::hex << xt0 << std::setfill('0') << ENDL;
    for (int i=0; i < DIDX; i++) {
        Code &c = DICT(i);
        IU  ip = c.udf ? c.pfa : (IU)(((UFP)c.xt & MSK_XT) - xt0);
        d2h_strcpy(name, (char*)c.name);
        fout << std::dec << std::setw(4) << i << '|'
             << std::hex << std::setw(3) << i << " :"
             << std::setw(6) << ip
             << (c.udf ? 'u' : ' ')
			 << (c.imm ? '*' : ' ') << ' '
             << name << std::endl;
    }
    fout << std::setbase(base) << std::setfill(' ') << std::setw(-1);
}

__HOST__ void Debug::self_tests() {
//    dict_dump(10);
//    words();
//    mem_dump(0, 256, 10);
    ss_dump(0, 3, 10);
}
