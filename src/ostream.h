/*! @file
  @brief
  cueForth Ostream module.

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_OSTREAM_H_
#define CUEF_SRC_OSTREAM_H_
#include "cuef_config.h"
#include "cuef_types.h"

#ifdef CUEF_USE_STRING
#include "string.h"
#endif // CUEF_USE_STRING

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
    U32 id   : 12;
    GT  gt   : 4;
    U32 size : 16;
    U8  data[];      // different from *data
} obuf_node;

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
/// Ostream class
///
class Ostream
{
    char *_buf = NULL;
    char *_ptr = 0;
    int  _sz   = 0;
    int  _base = 10;
    int  _width= 6;
    char _fill = ' ';
    int  _prec = 6;

    __GPU__  void _write(GT gt, U8 *v, int sz) {
        if (threadIdx.x!=0) return;                     // only thread 0 within a block can write
        if ((sizeof(obuf_node)+ALIGN4(sz))>size()) return; // too big

        //_LOCK;
        obuf_node *n = (obuf_node *)_ptr;

        n->id   = blockIdx.x;               // VM.id
        n->gt   = gt;
        n->size = ALIGN4(sz);               // 32-bit alignment

        MEMCPY(n->data, v, sz);

        _ptr  = ((char*)(n->data))+n->size;// advance pointer to next print block
        *_ptr = (char)GT_EMPTY;
        //_UNLOCK;
    }

public:
    ///
    /// constructor with managed memory buffer
    ///
    __GPU__  Ostream(char *buf, int sz=CUEF_OBUF_SIZE) : _buf(buf), _ptr(buf), _sz(sz) {}
    ///
    /// clear output buffer
    ///
    __GPU__  Ostream& clear() { _ptr = _buf; return *this; }
    __GPU__  U32 tellp()      { return (U32)(_ptr - _buf); }
    __GPU__  U32 size()       { return (U32)_sz; }
    ///
    /// iomanip control
    ///
    __GPU__ Ostream& operator<<(_setbase b) { _base  = b.base;  return *this; }
    __GPU__ Ostream& operator<<(_setw    w) { _width = w.width; return *this; }
    __GPU__ Ostream& operator<<(_setfill f) { _fill  = f.fill;  return *this; }
    __GPU__ Ostream& operator<<(_setprec p) { _prec  = p.prec;  return *this; }
    ///
    /// object input
    ///
    __GPU__ Ostream& operator<<(U8 c) {
        U8 buf[2] = { c, '\0' };
        _write(GT_STR, buf, 2);
        return *this;
    }
    __GPU__ Ostream& operator<<(GI i) {
        _write(_base==10 ? GT_INT : GT_HEX, (U8*)&i, sizeof(GI));
        return *this;
    }
    __GPU__ Ostream& operator<<(GF f) {
        _write(GT_FLOAT, (U8*)&f, sizeof(GF));
        return *this;
    }
    __GPU__ Ostream& operator<<(const char *s) {
        int i = STRLENB(s)+1;
        _write(GT_STR, (U8*)s, i);
        return *this;
    }
#if CUEF_USE_STRING
    __GPU__ Ostream& operator<<(string &s) {
        _write(GT_STR, (U8*)s.c_str(), s.size()+1);
        return *this;
    }
#endif // CUEF_USE_STRING
};
#endif // CUEF_SRC_OSTREAM_H_
