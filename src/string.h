#ifndef __EFORTH_SRC_STRING_H
#define __EFORTH_SRC_STRING_H
#include "util.h"

#define STRING_BUF_SIZE  16

namespace cuef {
///
/// stringbuf class
///
struct string : public vector<char>
{
	__GPU__ string(int asz=STRING_BUF_SIZE) {
		n = 0; if (asz>0) v = (char*)malloc(sz=asz);
	}
	__GPU__ string(const char *s, int asz=STRING_BUF_SIZE) {
		n  = STRLENB(s)+1;								// '\0'
		sz = ALIGN4(asz>n ? asz : n);
		v  = (char*)malloc(sz);
		MEMCPY(v, s, n);
	}
	__GPU__ ~string() { if (v) free(v); }

	__GPU__ string& operator<<(string& s)    { merge(s.v, s.size());        return *this; }
	__GPU__ string& operator<<(const char *s){ merge((char*)s, STRLENB(s)); return *this; }
	__GPU__ string& operator<<(int i) {
		string s(16);
		//TODO
		return *this;
	}
	__GPU__ string& operator<<(float v)      {
		string s(16);
		return *this;
	}
	__GPU__ bool    operator==(string& s)    { return STRCMP(v, s.v)==0; }
	__GPU__ string& substr(int i)    { string *s = new string(&v[i], n-i); return *s; }
	__GPU__ const char *c_str() { return v; }
	__GPU__ int  to_i(char **p, int base=10) {
        char c, *s = v;
        int  acc=0, neg = 0;
        *p = '\0';
        // handle leading blank and +/- sign
        do { c = *s++; } while (c==' ' || c=='\t');
        if (c == '-') { neg = 1; c = *s++; }
        else if (c == '+') c = *s++;
        // handle hex number
        if (c == '0' && (*s == 'x' || *s == 'X')) {
            if (base == 16) {
                c = s[1];
                s += 2;
                base = 16;
            }
            else return 0;
        }
        // process string
        while (c=*s++) {
            if (c>='0' && c<='9') c -= '0';
            else if ((c&0x5f) >= 'A') c = (c&0x5f) - 'A';
            else if (c >= base) break;
            *p  =  s;
            acc *= base;
            acc += c;
        }
        return neg ? -acc : acc;
 	}
};
    
} // namespace cuef
#endif // __EFORTH_SRC_STRING_H
