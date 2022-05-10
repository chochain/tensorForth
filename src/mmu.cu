/*! @file
  @brief
  cueForth - Memory Manager
*/
#include <iomanip>          // setw, setbase
#include "mmu.h"

__HOST__
MMU::MMU() {
    cudaMallocManaged(&_dict, sizeof(Code) * CUEF_DICT_SZ);
    cudaMallocManaged(&_pmem, sizeof(U8) * CUEF_HEAP_SZ);
    cudaMallocManaged(&_vss,  sizeof(DU) * CUEF_SS_SZ * MIN_VM_COUNT);
    GPU_CHK();
    printf("H: dict=%p, mem=%p, vss=%p\n", _dict, _pmem, _vss);
}
__HOST__
MMU::~MMU() {
    GPU_SYNC();
    cudaFree(_vss);
    cudaFree(_pmem);
    cudaFree(_dict);
}
///
/// dictionary search functions - can be adapted for ROM+RAM
///
__GPU__ int
MMU::find(const char *s, bool compile, bool ucase) {
    printf("find(%s) => ", s);
    for (int i = _didx - (compile ? 2 : 1); i >= 0; --i) {
        const char *t = _dict[i].name;
        if (ucase && STRCASECMP(t, s)==0) return i;
        if (!ucase && STRCMP(t, s)==0) return i;
    }
    return -1;
}
///
/// colon - dictionary word compiler
///
__GPU__ void
MMU::colon(const char *name) {
    int  sz = STRLENB(name);                // aligned string length
    Code *c = &_dict[_didx++];              // get next dictionary slot
    align();                                // nfa 32-bit aligned (adjust _midx)
    c->name = (const char*)&_pmem[_midx];   // assign name field index
    c->def  = 1;                            // specify a colon word
    c->nlen = sz;                           // word name length (for colon word only)
    add((U8*)name,  ALIGN2(sz+1));          // setup raw name field
    c->plen = 0;                            // reset parameter counter (by number of U16)
    c->pidx = _midx;                        // capture code field index
}
///
/// Debugging methods
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ void
MMU::to_s(std::ostream &fout, IU w) {
    /*
     * TODO: not sure why copying 32 byt does not work?
     * char name[36];
     * cudaMemcpy(name, _dict[w].name, 32, D2H);
     */
    U8 c, i=0;
    cudaMemcpy(&c, _dict[w].name, 1, D2H);
    while (c) {
        fout << c;
        cudaMemcpy(&c, _dict[w].name+(++i), 1, D2H);
    }
    fout << " " << w << (_dict[w].immd ? "* " : " ");
}
///
/// display dictionary word list
///
__HOST__ void
MMU::words(std::ostream &fout) {
    fout << std::setbase(10);
    for (int i=0; i<_didx; i++) {
        if ((i%10)==0) { fout << std::endl; }
        to_s(fout, i);
    }
}
///
/// recursively disassemble colon word
///
__HOST__ void
MMU::see(std::ostream &fout, U8 *wp, int *i, int level) {
    IU    w = ri((IU*)wp);
    Code *c = &_dict[w];
    fout << std::endl; for (int n=level; n>0; n--) fout << "  ";    // indentation by level
    if (level) fout << "[" << std::setw(4) << *i << ": ";
    else       fout << "[ " << (void*)wp << ": ";
    to_s(fout, w);                                                  // display word name
    if (c->def && level==0) {                                       // is a colon word?
        int i1 = 0;                                                 // display children recursively
        U8  *p = pfa(w);
        while (i1 < c->plen) {
            see(fout, p + i1, &i1, level+1);                        // recursive call
        }
    }
    wp += sizeof(IU);                                               // advance word pointer
    *i += sizeof(IU);                                               // advance IP
    switch (w) {
    case DOVAR: case DOLIT:
        fout << "= " << rd((DU*)wp); *i += sizeof(DU); break;       // fetch literal
    case DOSTR: case DOTSTR: {
        char *s = (char*)wp;
        int  sz = strlen(s)+1;
        *i += ALIGN2(sz);                                           // fetch string
        fout << "= \"" << s << "\"";
    } break;
    case BRAN: case ZBRAN: case DONEXT:
        fout << "j" << ri((IU*)wp); *i += sizeof(IU); break;        // fetch jump target
    }
    fout << "] ";
}
__HOST__ void
MMU::see(std::ostream &fout, IU w) {
    int i = 0;
    see(fout, (U8*)&w, &i, 0);
}
///
/// dump data stack content
///
__HOST__ void
MMU::ss_dump(std::ostream &fout, int vid, int n) {
    DU *ss = &_vss[vid * CUEF_SS_SZ];
    fout << " <";
    for (int i=0; i<n; i++) { fout << ss[i] << " "; }
    fout << ss[CUEF_SS_SZ-1] << "> ok" << std::endl;
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
MMU::dump(std::ostream &fout, IU p0, int sz) {
    const char *i2h = "0123456789abcdef";
    char buf[80];
    for (IU i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (int j=0; j<16; j++) {
            //U8 c = *(((U8*)&_dict[0])+i+j) & 0x7f;               // to dump _dict
            U8 c = _pmem[i+j] & 0x7f;
            C2H(c);                                                // "%02x "
            buf[x++] = ' ';
            if (j%4==3) buf[x++] = ' ';
            buf[59+j]= (c==0x7f||c<0x20) ? '.' : c;                // %c
        }
        buf[75] = '\0';
        fout << buf;
    }
    fout << std::endl;
}
