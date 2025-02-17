/** -*- c++ -*-
 * @file
 * @brief System class - tensorForth Debug/Tracer implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>
#include "debug.h"

#define MEM(a)  ((U8*)&mu->_pmem[a])
#define DICT(w) (mu->_dict[w])
#define DIDX    (mu->_didx)
///
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
    DU *ss = mu->vmss(id);                ///< retrieve VM SS
    h_ostr &fout = io->fout;
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
    for (int i=0; i<n; i++) {
        fout << rdx(*ss++, base) << ' ';
    }
    fout << "-> ok" << ENDL;
}
__HOST__ int
Debug::p2didx(Param *p) {
    IU pfa = p->ioff;
    for (int i = DIDX - 1; i >= 0; --i) {
        Code &c = DICT(i);
        if (p->udf) { if (c.udf && pfa==c.pfa) return i; }
        else        { if (!c.udf && pfa==mu->H_XTOFF(c.xt)) return i; }
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
    
    fout << ENDL; fout << "  ";                    /// * indent
    if (io->trace) {                               /// * header
        fout << std::setbase((int)16) << "( ";
        fout << std::setfill('0') << std::setw(4) << ((U8*)p - MEM(0)); ///> addr
        fout << '['   << std::setfill(' ') << std::setw(4) << w << ']'; ///> word ref
        fout << " ) " << std::setbase(base);
    }
    if (!pm) {                                     ///> built-in
        U8 name[36], i=0;                          ///< name buffer on host
//        cudaMemcpy(name, code.name, 1, D2H);
        do {
            cudaMemcpy(name+i, code.name+i, 1, D2H);
        } while (name[i++] != '\0');
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
        fout << " \\ $" << std::setbase(16)
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
    fout << std::setbase(10);
    for (int i=0, sz=0; i < DIDX; i++) {
        const char *nm = DICT(i).name;
        const int  len = strlen(nm);
#if CC_DEBUG > 1
        if (nm[0])
#else  //  CC_DEBUG > 1
        if (nm[len-1] != ' ')
#endif // CC_DEBUG > 1
        {            
            sz += len + 2;
            fout << "  " << nm;
        }
        if (io->trace || sz > WIDTH) {
            fout << ENDL; sz = 0;
        }
    }
    if (!io->trace) fout << std::setbase(base) << ENDL;
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
    fout << std::setbase(16) << std::setfill('0');
    for (IU i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (IU j=0; j<16; j++) {
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
    fout << std::setbase(base) << std::setfill(' ');
}

#define NFA(w) (DICT(w).pfa - ALIGN(strlen(DICT(w).name)))
__HOST__ void
Debug::see(IU w, int base) {
    h_ostr &fout = io->fout;
    Code   &c    = DICT(w);
    fout << ": " << c.name << ENDL;
    if (!c.udf) {
        fout << " ( built-ins ) ;" << ENDL;
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
    fout << ENDL;
}
///====================================================================
///
///> System statistics - for heap, stack, external memory debugging
///
__HOST__ void
Debug::dict_dump(int base) {
    h_ostr &fout = io->fout;
    fout << std::setbase(16) << std::setfill('0') << ENDL;
    for (int i=0; i < DIDX; i++) {
        Code &c = DICT(i);
        fout << std::setfill('0') << std::setw(3) << i
             << (c.udf ? " U" : "  ")
			 << (c.imm ? "I " : "  ")
             << std::setw(8) << (UFP)c.xt
             << ":" << std::setw(6) << c.pfa
             << " " << c.name << ENDL;
    }
    fout << std::setbase(base) << std::setfill(' ') << std::setw(-1);
}
