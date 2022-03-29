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
#include "util.h"

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
    GT  gt   : 4;
    U32 id   : 12;
    U32 size : 16;
    U8  data[];      // different from *data
} obuf_node;
#define NODE_SZ  sizeof(U32)

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
#include <stdio.h>
class Ostream : public Managed {
    char *_buf;
    int  _max  = 0;
    int  _idx  = 0;
    int  _base = 10;
    int  _width= 6;
    char _fill = ' ';
    int  _prec = 6;

    __GPU__ void _dump() {
        for (int i=0; i<=_idx; i++) {
        	printf("%02x %c ", _buf[i], _buf[i] < 0x20 ? '.' : _buf[i]);
        }
        printf("%c", '\n');

    }
    __GPU__  void _write(GT gt, U8 *v, int sz) {
        if (threadIdx.x!=0) return;                                 // only thread 0 within a block can write

        //_LOCK;
        obuf_node *n = (obuf_node*)&_buf[_idx];                     // allocate next node

        n->gt   = gt;                // data type
        n->id   = blockIdx.x;        // VM.id
        n->size = ALIGN4(sz);        // 32-bit alignment

        int inc = NODE_SZ + n->size; // calc node allocation size

        printf("_idx %d += %d\n", _idx, inc);

        if ((_idx + inc) > _max) inc = 0;     // overflow, skip
        else MEMCPY(n->data, v, sz);          // deep copy, TODO: shallow copy via managed memory

        _buf[(_idx += inc)] = (char)GT_EMPTY; // advance index and mark end of stream
        //_UNLOCK;
        //_dump();
    }

public:
    Ostream(int sz=CUEF_OBUF_SIZE) { cudaMallocManaged(&_buf, _max=sz); GPU_CHK(); }
    ~Ostream()                     { GPU_SYNC(); cudaFree(_buf); }
    ///
    /// clear output buffer
    ///
    __HOST__ Ostream& clear() {
    	// LOCK
    	_buf[_idx=0] = (char)GT_EMPTY;
    	// UNLOCK
    	return *this;
    }
    __HOST__ char *rdbuf() { return _buf; }
    __HOST__ U32 tellp()   { return (U32)_idx; }
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
};
#endif // CUEF_SRC_OSTREAM_H_
