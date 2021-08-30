#ifndef __EFORTH_SRC_STRING_H
#define __EFORTH_SRC_STRING_H
#include "vector.h"

#define STRING_BUF_SIZE  16

namespace cuef {
///
/// stringbuf class
///
struct string : public vector<char>
{
    ///
    /// constructors
    ///
	__GPU__ string(int asz=STRING_BUF_SIZE) {
		_n = 0; if (asz>0) _v = (char*)malloc(_sz=ALIGN4(asz));
	}
	__GPU__ string(const char *s, int asz=STRING_BUF_SIZE) {
		_n  = STRLENB(s);
		_sz = ALIGN4(asz>(_n+1) ? asz : (_n+1));
        _v  = (char*)malloc(_sz);
        MEMCPY(_v, s, _n+1);
	}
    ///
    /// string export
    ///
    __GPU__ string& str()         { _v[_n] = '\0'; return *this; }
	__GPU__ char    *c_str()      { return str()._v; }
	__GPU__ string& substr(int i) {
        string *s = new string(&_v[i], _n-i);
        return *s;
    }
    ///
    /// compare
    ///
	__GPU__ bool operator==(const char *s2) { return STRCMP(_v, s2)==0; }
	__GPU__ friend bool operator==(const string& s1, const string& s2) {
        return STRCMP(s1._v, s2._v)==0;
    }
    ///
    /// assignment
    ///
	__GPU__ string& operator<<(const char *s) { merge((char*)s, STRLENB(s)); return str(); }
	__GPU__ string& operator<<(string& s)     { merge(s._v, s.size());       return str(); }
	__GPU__ string& operator<<(int i) {
        char s[20];
        ITOA(i, s, 10);
        merge(s, STRLENB(s));
        return str(); 
    }
	__GPU__ string& operator<<(float f) {
        char s[20];
		if (f < 0) { f = -f; push('-'); }
		int i = static_cast<int>(f);
        int d = static_cast<int>(round(1000000*(f - i)));
		ITOA(i, s, 10);
        merge(s, STRLENB(s));
		push('.');
        ITOA(d, s, 10);
        merge(s, STRLENB(s));
        return str();
	}
    ///
    /// conversion
    ///
	__GPU__ int   to_i(char **p, int base=10) { return (int)STRTOL(_v, p, base); }
    __GPU__ float to_f(char **p)              { return (float)STRTOF(_v, p);     }
};
    
} // namespace cuef
#endif // __EFORTH_SRC_STRING_H
