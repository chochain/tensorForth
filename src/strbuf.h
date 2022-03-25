/*! @file
  @brief
  cueForth string buffer class

  <pre>
  Copyright (C) 2022- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_STRBUF_H
#define CUEF_SRC_STRBUF_H
#include "vector.h"
#include <stdio.h>

#define CUEF_FLOAT_PRECISION  1000000     /* 6-digit */
#define STRBUF_SIZE           8
///
/// string buffer class
///
struct StrBuf : public Vector<char>
{
    ///
    /// constructors
    ///
    __GPU__ StrBuf(int asz=STRBUF_SIZE) {
        if (asz>0) v = new char[max=ALIGN4(asz)];
    }
    __GPU__ StrBuf(const char *s) {
        idx = STRLENB(s);
        v = new char[max=ALIGN4(idx+1)];
        MEMCPY(v, s, idx);
    }
    __GPU__ StrBuf(StrBuf& s) {
        idx = s.idx;
        v = new char[max=ALIGN4(idx+1)];
        MEMCPY(v, s.c_str(), idx);
    }
    ///
    /// StrBuf export
    ///
    __GPU__ __INLINE__ StrBuf& str() { v[idx]='\0'; return *this; }
    __GPU__ __INLINE__ char *c_str() { v[idx]='\0'; return v;    }
    __GPU__ StrBuf& substr(int i) {
        v[idx] = '\0';
        StrBuf& s = *new StrBuf(&v[i]);
        return s;
    }
    ///
    /// compare
    ///
    __GPU__ bool operator==(const char *s2)   { return memcmp(v, s2, idx)==0; }
    __GPU__ bool operator==(const StrBuf& s2) { return memcmp(v, s2.v, idx)==0; }
    ///
    /// assignment
    ///
    __GPU__ StrBuf& operator<<(const char c)  { push(c); return *this; }
    __GPU__ StrBuf& operator<<(const char *s) { merge((char*)s, STRLENB(s)); return *this; }
    __GPU__ StrBuf& operator<<(StrBuf& s)     { merge(s.v, s.idx); return *this; }
    __GPU__ StrBuf& operator<<(int i) {
        char s[36];
        int n = ITOA(i, s, 10);
        merge(s, n);
        return *this;
    }
    __GPU__ StrBuf& operator<<(float f) {
        if (f < 0) { f = -f; push('-'); }
        int i = static_cast<int>(f);
        int d = static_cast<int>(round(CUEF_FLOAT_PRECISION*(f - i)));
        return *this << i << '.' << d;
    }
    ///
    /// conversion
    ///
    __GPU__ int   to_i(char **p, int base=10) { return (int)STRTOL((char*)v, p, base); }
    __GPU__ float to_f(char **p)              { return (float)STRTOF((char*)v, p);     }
};
#endif // CUEF_SRC_STRBUF_H
