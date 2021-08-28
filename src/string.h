#ifndef __EFORTH_SRC_STRING_H
#define __EFORTH_SRC_STRING_H
#include "util.h"

namespace cuef {
///
/// stringbuf class
///
struct string : public vector<char>
{
	__GPU__ string(int asz=16) {
		n = 0; if (asz>0) v = (char*)malloc(sz=asz);
	}
	__GPU__ string(const char *s, int asz=16) {
		n  = STRLENB(s)+1;								// '\0'
		sz = ALIGN4(asz>n ? asz : n);
		v  = (char*)malloc(sz);
		MEMCPY(v, s, n);
	}
	__GPU__ ~string() { if (v) free(v); }

	__GPU__ string& operator<<(string s)     { merge(s.v, s.size()+1);        return *this; }
	__GPU__ string& operator<<(const char *s){ merge((char*)s, STRLENB(s)+1); return *this; }
	__GPU__ bool    operator==(string s)     { return STRCMP(v, s.v); }
	__GPU__ string  substr(int i)            { int m = n-i+1; string s(m); MEMCPY(s.v, &v[i], m); s.n = n-i; return s; }
	__GPU__ char *c_str() { return v; }
};
    
} // namespace cuef
#endif // __EFORTH_SRC_STRING_H
