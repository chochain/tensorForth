/*! @file
  @brief
  cueForth - dictionary manager
*/
#include <iomanip>          // setw, setbase
#include "dict.h"

Dict::Dict() {
    cudaMallocManaged(&_dict, sizeof(Code)*CUEF_DICT_SZ);
    cudaMallocManaged(&_pmem, sizeof(U8)*CUEF_HEAP_SZ);
    GPU_CHK();
}
Dict::~Dict() {
    GPU_SYNC();
    cudaFree(_pmem);
    cudaFree(_dict);
}
///
/// dictionary search functions - can be adapted for ROM+RAM
///
__GPU__ int
Dict::find(const char *s, bool compile, bool ucase) {
	printf("find(%s) => ", s);
    for (int i = _didx - (compile ? 2 : 1); i >= 0; --i) {
    	const char *t = _dict[i].name;
    	if (ucase && STRCASECMP(t, s)==0) return i;
    	if (!ucase && STRCMP(t, s)==0) return i;
    }
    return -1;
}
///
/// compiler
///
__GPU__ void
Dict::colon(const char *name) {
    int  sz = STRLENB(name);                // aligned string length
    Code *c = &_dict[_didx++];              // get next dictionary slot
	align();                                // nfa 32-bit aligned (adjust _midx)
    c->name = (const char*)&_pmem[_midx];   // assign name field index
    c->def  = 1;                            // specify a colon word
    c->nlen = sz;                           // word name length (for colon word only)
    c->plen = 0;                            // advance counter (by number of U16)
    add((U8*)name,  ALIGN2(sz+1));          // setup raw name field
    c->pidx = _midx;                        // capture code field index
}
///
/// Debugging methods
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ void
Dict::to_s(std::ostream &fout, IU w) {
	/*
	 * TODO: not sure why copying 32 byt does not work?
	 * U8 name[36];
	 * cudaMemcpy(&name[0], _dict[w].name, 32, cudaMemcpyDeviceToHost);
	 */
	U8 c, i=0;
	cudaMemcpy(&c, _dict[w].name, 1, cudaMemcpyDeviceToHost);
	while (c) {
		fout << c;
		cudaMemcpy(&c, _dict[w].name+(++i), 1, cudaMemcpyDeviceToHost);
	}
    fout << " " << w << (_dict[w].immd ? "* " : " ");
}
///
/// recursively disassemble colon word
///
__HOST__ void
Dict::see(std::ostream &fout, IU *cp, IU *ip, int dp) {
    IU   c  = ri(cp);
    Code *w = &_dict[c];
    fout << std::endl; for (int i=dp; i>0; i--) fout << "  ";       // indentation
    if (dp) fout << "[" << std::setw(2) << ri(ip) << ": ";          // ip offset
    else    fout << "[ ";
    to_s(fout, c);
    if (w->def) {                                                   // a colon word
        for (IU ip1=0, n=w->plen; ip1<n; ip1+=sizeof(IU)) {         // walk through children
            IU *cp1 = (IU*)(pfa(c) + ip1);                          // next children node
            see(fout, cp1, &ip1, dp+1);                             // dive recursively
        }
    }
    switch (c) {
    case DOVAR: case DOLIT:
        fout << "= " << rd((DU*)(cp+1)); *ip += sizeof(DU); break;
    case DOSTR: case DOTSTR: {
    	char *s = (char*)(cp+1);
    	int  sz = strlen(s)+1;
        *ip += ALIGN2(sz);                                           // advance IP
        fout << "= \"" << s << "\"";
    } break;
    case BRAN: case ZBRAN: case DONEXT:
        fout << "j" << ri(cp+1); *ip += sizeof(IU); break;
    }
    fout << "] ";
}
///
/// display dictionary word list
///
__HOST__ void
Dict::words(std::ostream &fout) {
	fout << std::setbase(10);
    for (int i=0; i<_didx; i++) {
        if ((i%10)==0) { fout << std::endl; }
        to_s(fout, i);
    }
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
Dict::dump(std::ostream &fout, IU p0, int sz) {
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
}
