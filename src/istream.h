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
#include "cuef_config.h"
#include "cuef_types.h"
#include "util.h"
///
/// istream class
///
#include <stdio.h>
class Istream : public Managed {
    char *_buf;                 /// input buffer
    int  _idx  = 0;             /// current buffer index
    int  _gn   = 0;             /// number of byte processed

#if CUEF_DEBUG
    __GPU__ __INLINE__ void _debug() { printf("%d>> ibuf[%d] >> %d bytes\n", blockIdx.x, _idx, _gn); }
#else
    __GPU__ __INLINE__ void _debug() {}
#endif // CUEF_DEBUG

    __GPU__ int _tok(char delim) {
        char *p = &_buf[_idx];
        while (delim==' ' && (*p==' ' || *p=='\t')) (p++, _idx++); // skip leading blanks and tabs
        int nidx=_idx; while (*p && *p!=delim) (p++, nidx++);      // advance pointers
        _gn = (delim!=' ' && *p!=delim) ? nidx=0 : nidx - _idx;    // not found or end of input string
        return nidx;
    }
public:
    Istream(int sz=CUEF_IBUF_SIZE) { cudaMallocManaged(&_buf, sz);	GPU_CHK(); }
    ~Istream()                     { GPU_SYNC(); cudaFree(_buf); }
    ///
    /// intialize by a given string
    ///
    __HOST__ char     *rdbuf() { return _buf; }
    __HOST__ Istream& clear()  {
    	//LOCK;
    	_idx = _gn = 0;
    	//UNLOCK;
    	return *this;
    }
    __HOST__ Istream& str(const char *s, int sz=0) {
        memcpy(_buf, s, sz ? sz : strlen(s));
        _idx = 0;
        return *this;
    }
    ///
    /// sizing
    ///
    __HOST__ int gcount() { return _gn;  }
    __HOST__ int tellg()  { return _idx; }
    //
    /// parser
    ///
    __GPU__ Istream& get_idiom(char *s, char delim=' ') {
        int nidx = _tok(delim);             // index to next token

        _debug();

        if (nidx==0) return *this;          // no token processed
        memcpy(s, &_buf[_idx], _gn);        // CUDA memcpy
        s[_gn] = '\0';                      // terminated with '\0'
        _idx = nidx;                        // advance index
        return *this;
    }
    __GPU__ Istream& getline(char *s, int sz, char delim='\n') {
    	return get_idiom(s, delim);
    }
    __GPU__ int operator>>(char *s)   { get_idiom(s); return _gn; }
};
#endif // CUEF_SRC_ISTREAM_H_
