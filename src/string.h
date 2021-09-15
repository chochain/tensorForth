#ifndef __EFORTH_SRC_STRING_H
#define __EFORTH_SRC_STRING_H
#include "vector.h"
#include <stdio.h>

#define STRING_BUF_SIZE  8

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
		if (asz>0) _v = new char[_sz=ALIGN4(asz)];
	}
    __GPU__ string(const char *s) {
        _n = STRLENB(s);
        _v = new char[_sz=ALIGN4(_n+1)];
        MEMCPY(_v, s, _n);
    }
    __GPU__ string(string& s) {
        _n = s._n;
        _v = new char[_sz=ALIGN4(_n+1)];
        MEMCPY(_v, s.c_str(), _n);
    }
    ///
    /// string export
    ///
    __GPU__ __INLINE__ string& str() { _v[_n]='\0'; return *this; }
	__GPU__ __INLINE__ char *c_str() { _v[_n]='\0'; return _v;    }
	__GPU__ string& substr(int i) {
        _v[_n] = '\0';
        string& s = *new string(&_v[i]);
        return s;
    }
    ///
    /// compare
    ///
	__GPU__ bool operator==(const char *s2)   { return memcmp(_v, s2, _n)==0; }
	__GPU__ bool operator==(const string& s2) { return memcmp(_v, s2._v, _n)==0; }
    ///
    /// assignment
    ///
	__GPU__ string& operator<<(const char *s) { merge((char*)s, STRLENB(s)); return *this; }
	__GPU__ string& operator<<(string& s)     { merge(s._v, s._n); return *this; }
	__GPU__ string& operator<<(int i) {
        char s[36];
        int n = ITOA(i, s, 10);
        merge(s, n);
        return *this;
    }
	__GPU__ string& operator<<(float f) {
		if (f < 0) { f = -f; push('-'); }
		int i = static_cast<int>(f);
        int d = static_cast<int>(round(1000000*(f - i)));
        *this << i;
        push('.');
        *this << d;
        return *this;
	}
    ///
    /// conversion
    ///
	__GPU__ int   to_i(char **p, int base=10) { return (int)STRTOL(_v, p, base); }
    __GPU__ float to_f(char **p)              { return (float)STRTOF(_v, p);     }
};
    
} // namespace cuef
#endif // __EFORTH_SRC_STRING_H
