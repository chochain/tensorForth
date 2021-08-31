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
#include "string.h"

namespace cuef {
///
/// istream class
///
class istream
{
	char *_buf = NULL;  /// input buffer
	int  _idx  = 0;     /// current buffer index
    int  _max  = 0;     /// max length of input buffer
    int  _sz   = 0;     /// size processed
    
    __GPU__ int _tok(char delim) {
        while (delim==' ' &&
               (_buf[_idx]==' ' || _buf[_idx]=='\t')) _idx++; // skip leading blanks and tabs
        int i = _idx;  while (i<_max && _buf[i]!=delim) i++;
        int x = i>=_max;
        _sz = (delim!=' ' && x) ? i=0 : i - _idx + x;
        return i;
    }
public:
    __GPU__  istream(char *buf=NULL) : _buf(buf) {}
    ///
    /// intialize
    ///
    __GPU__  istream& str(const char *s, int sz=0) {
        _sz  = _max = sz ? sz : STRLENB(s);
        _buf = (char*)s;
        _idx = 0;
        return *this;
    }
    __GPU__  istream& str(string& s) {
        str(s.c_str(), s.size()); return *this;
    }
    ///
    /// sizing
    ///
    __GPU__ int size()   { return _max; }
    __GPU__ int gcount() { return _sz;  }
    //
    /// parser
    ///
    __GPU__ istream& getline(char *s, char delim=' ') {
        int i = _tok(delim); if (i==0) return *this;
        MEMCPY(s, &_buf[_idx], _sz);
        s[_sz+1] = '\0';                         // terminated with '\0'
        _idx = i;
        return *this;
    }
    __GPU__ istream& getline(string& s, char delim=' ') {
        int i = _tok(delim); if (i==0) return *this;
        s._n = 0;
        s.merge(&_buf[_idx], _sz);
        s._v[_sz+1] = '\0';
        _idx = i;
        return *this;
    }
    __GPU__  int operator>>(char *s)   { getline(s); return _sz; }
    __GPU__  int operator>>(string& s) { getline(s); return _sz; }
};

} // namespace cuef
#endif // CUEF_SRC_ISTREAM_H_
