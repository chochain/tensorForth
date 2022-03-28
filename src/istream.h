/*! @file
  @brief
  cueForth istream module.

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_ISTREAM_H_
#define CUEF_SRC_ISTREAM_H_
#include "util.h"
///
/// istream class
///
class Istream {
public:
    char *_buf = NULL;  /// input buffer
    int  _idx  = 0;     /// current buffer index
    int  _gn   = 0;     /// number of byte processed

    __GPU__ int _tok(char delim) {
        char *c = &_buf[_idx];
        while (delim==' ' && (*c==' ' || *c=='\t')) (c++, _idx++); // skip leading blanks and tabs
        int nidx=_idx; while (*c && *c!=delim) (c++, nidx++);
        _gn = (delim!=' ' && *c!=delim) ? nidx=0 : nidx - _idx;    // not found or end of input string
        return nidx;
    }

    __GPU__ Istream(char *s)  { _buf = s; }
    __GPU__ Istream(int sz=0) { if (sz) _buf = new char[_gn=ALIGN4(sz)]; }
    __GPU__ ~Istream()        { if (_buf) delete[] _buf; }
    ///
    /// intialize by a given string
    ///
    __GPU__ Istream& str(const char *s, int sz=0) {
        if (_buf) delete[] _buf;
        _buf = new char[_gn = ALIGN4(sz ? sz : STRLENB(s))];
        MEMCPY(_buf, s, _gn);
        _idx = 0;
        return *this;
    }
    ///
    /// sizing
    ///
    __GPU__ int gcount() { return _gn;  }
    __GPU__ int tellg()  { return _idx; }
    //
    /// parser
    ///
    __GPU__ Istream& get_idiom(char *s, char delim=' ') {
        int nidx = _tok(delim);             // index to next token
        if (nidx==0) return *this;          // no token processed
        MEMCPY(s, &_buf[_idx], _gn);
        s[_gn + 1] = '\0';                  // terminated with '\0'
        _idx = nidx;                        // advance index
        return *this;
    }
    __GPU__ Istream& getline(char *s, int sz, char delim='\n') {
    	return get_idiom(s, delim);
    }
    __GPU__ int operator>>(char *s)   { get_idiom(s); return _gn; }

#if CUEF_USE_STRBUF
#include "strbuf.h"
    __GPU__  Istream& str(string& s) {
        str(s.c_str(), s.size()); return *this;
    }
    __GPU__ Istream& get_idiom(string& s, char delim=' ') {
        int nidx = _tok(delim);
        if (nidx==0) return *this;
        s._n = 0;
        s.merge(&_buf[_idx], _gn);
        s._v[_gn + 1] = '\0';
        _idx = nidx;
        return *this;
    }
    __GPU__  int operator>>(string& s) { get_idiom(s); return _gn; }
#endif // CUEF_USE_STRBUF
};
#endif // CUEF_SRC_ISTREAM_H_
