/*! @file
  @brief
  cueForth string stream module.

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_SSTREAM_H_
#define CUEF_SRC_SSTREAM_H_
#include "string.h"

typedef uint32_t  U32;
typedef uint8_t   U8;

//================================================================
/*!@brief
  define the value type.
*/
typedef enum {
    GT_EMPTY = 0,
    GT_INT,
    GT_HEX,
    GT_FLOAT,
    GT_STR,
} GT;

//================================================================
/*! printf internal version data container.
*/
typedef struct {
	U32	id   : 12;
    GT  gt 	 : 4;
    U32	size : 16;
    U8	data[];          								// different from *data
} print_node;

namespace cuef {
///
/// istream class
///
class istream
{
	char *_buf = NULL;  /// input buffer
    int  _max  = 0;     /// max length of input buffer
	int  _idx  = 0;     /// current buffer index
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
///
/// iomanip classes
///
class _setbase { public: int  base;  __GPU__ _setbase(int b) : base(b)  {}};
class _setw    { public: int  width; __GPU__ _setw(int w)    : width(w) {}};
class _setfill { public: char fill;  __GPU__ _setfill(char f): fill(f)  {}};
class _setprec { public: int  prec;  __GPU__ _setprec(int p) : prec(p)  {}};
__GPU__ __INLINE__ _setbase setbase(int b)  { return _setbase(b); }
__GPU__ __INLINE__ _setw    setw(int w)     { return _setw(w);    }
__GPU__ __INLINE__ _setfill setfill(char f) { return _setfill(f); }
__GPU__ __INLINE__ _setprec setprec(int p)  { return _setprec(p); }
///
/// ostream class
///
class ostream
{
	U8   *_buf = NULL;
	int  _sz   = 0;
	int  _base = 10;
	int  _width= 6;
	char _fill = ' ';
	int  _prec = 6;

    __GPU__  void _write(GT gt, U8 *v, int sz);
    
public:
    __GPU__  ostream(U8 *buf, int sz=CUEF_OBUF_SIZE) : _buf(buf), _sz(sz) {}
    ///
    /// iomanip control
    ///
    __GPU__ ostream& operator<<(_setbase b) { _base  = b.base;  return *this; }
    __GPU__ ostream& operator<<(_setw    w) { _width = w.width; return *this; }
    __GPU__ ostream& operator<<(_setfill f) { _fill  = f.fill;  return *this; }
    __GPU__ ostream& operator<<(_setprec p) { _prec  = p.prec;  return *this; }
    ///
    /// object input
    ///
    __GPU__ ostream& operator<<(U8 c) {
        U8 buf[2] = { c, '\0' };
        _write(GT_STR, buf, 2);
        return *this;
    }
    __GPU__ ostream& operator<<(GI i) {
        _write(_base==10 ? GT_INT : GT_HEX, (U8*)&i, sizeof(GI));
        return *this;
    }
    __GPU__ ostream& operator<<(GF f) {
        _write(GT_FLOAT, (U8*)&f, sizeof(GF));
        return *this;
    }
    __GPU__ ostream& operator<<(const char *s) {
        int i = STRLENB(s)+1;
        _write(GT_STR, (U8*)s, i);
        return *this;
    }
    __GPU__ ostream& operator<<(string &s) {
        U8 *s1 = (U8*)s.c_str();
        int i  = STRLENB(s1);
        _write(GT_STR, s1, i);
        return *this;
    }
};

}   // namespace cuef

// global output buffer for now, per session later
extern __GPU__ GI  _output_size;
extern __GPU__ U8  *_output_buf;
extern __GPU__ U8  *_output_ptr;

__KERN__ void        stream_init(U8 *buf, int sz);
__HOST__ print_node* stream_print(print_node *node, int trace);
__HOST__ void        stream_flush(U8 *output_buf, int trace);

#endif // CUEF_SRC_SSTREAM_H_
