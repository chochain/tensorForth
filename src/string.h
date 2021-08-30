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
	__GPU__ string& operator<<(int i) {	return to_string(i); }
	__GPU__ string& operator<<(float v)      {
		if (v < 0) { v = -v; push('-'); }
		int vi = static_cast<int>(v);
		to_string(vi);
		push('.');
		to_string(static_cast<int>(1000000*(v - vi)));
		return *this;
	}
	__GPU__ bool    operator==(string& s)    { return STRCMP(v, s.v)==0; }
	__GPU__ string& substr(int i)    { string *s = new string(&v[i], n-i); return *s; }
	__GPU__ string& to_string(int v, int base=10) {
	    int x = v;
	    if (x < 0) { x=-x; push('-'); }
	    do {
	        int dx = x % 10;
	        push((char)(dx+'0'));
	        x /= 10;
	    } while (x != 0);
	    return *this;
	}
	__GPU__ const char *c_str() { return v; }
	__GPU__ int   to_i(char **p, int base=10) { return (int)STRTOL(v, p, base); }
    __GPU__ float to_f(char **p)              { return (float)STRTOF(v, p);     }
};
    
} // namespace cuef
#endif // __EFORTH_SRC_STRING_H
