/*! @file
  @brief
  cueForth ostream module.

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_OSTREAM_H_
#define CUEF_SRC_OSTREAM_H_
#include "string.h"

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

#define _LOCK		{ MUTEX_LOCK(_mutex_ss); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_ss); }

__GPU__ volatile int _mutex_ss;
///
/// ostream class
///
class ostream
{
	char *_buf = NULL;
	char *_ptr = 0;
	int  _sz   = 0;
	int  _base = 10;
	int  _width= 6;
	char _fill = ' ';
	int  _prec = 6;

    __GPU__  void _write(GT gt, U8 *v, int sz) {
        if (threadIdx.x!=0) return;						// only thread 0 within a block can write

        _LOCK;
        print_node *n = (print_node *)_ptr;
        MEMCPY(n->data, v, sz);

        n->id   = blockIdx.x;							// VM.id
        n->gt   = gt;
        n->size = ALIGN4(sz);							// 32-bit alignment

        _ptr  = (char*)U8PADD(n->data, n->size);		// advance pointer to next print block
        *_ptr = (U8)GT_EMPTY;
        _UNLOCK;
    }        
    
public:
    __GPU__  ostream(char *buf, int sz=CUEF_OBUF_SIZE) : _buf(buf), _ptr(buf), _sz(sz) {}
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
#endif // CUEF_SRC_OSTREAM_H_
