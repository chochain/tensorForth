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
    GT_FLOAT,
    GT_STR,
    GT_FMT,
    GT_OPX
} GT;

typedef enum {
    OP_WORDS = 0,
    OP_SEE,
    OP_DUMP,
    OP_SS
} OP;

//================================================================
/*! printf internal version data container.
*/
typedef struct {
    U16 gt   : 4;
    U16 id   : 12;
    U16 sz;
    U8  data[];      // different from *data
} obuf_node;

typedef struct {
    U8 base;
    U8 width;
    U8 prec;
    U8 fill;
} obuf_fmt;

#define NODE_SZ  sizeof(U32)
///
/// implement kernel iomanip classes
///
struct _setbase { U8  base;  __GPU__ _setbase(int b) : base(b)  {}};
struct _setw    { U8  width; __GPU__ _setw(U8 w)    : width(w) {}};
struct _setfill { U8 fill;   __GPU__ _setfill(U8 f) : fill(f)  {}};
struct _setprec { U8  prec;  __GPU__ _setprec(U8 p) : prec(p)  {}};
__GPU__ __INLINE__ _setbase setbase(int b)  { return _setbase((U8)b); }
__GPU__ __INLINE__ _setw    setw(int w)     { return _setw((U8)w);    }
__GPU__ __INLINE__ _setfill setfill(char f) { return _setfill((U8)f); }
__GPU__ __INLINE__ _setprec setprec(int p)  { return _setprec((U8)p); }
///
/// Forth parameterized manipulators
///
struct _opx     { int op, a, n; __GPU__ _opx(int op, int a, int n) : op(op), a(a), n(n) {}};
__GPU__ __INLINE__ _opx opx(OP op, int a=0, int n=0) { return _opx(op, a, n); }
///
/// Ostream class
///
#include <stdio.h>
class Ostream : public Managed {
    char    *_buf;
    int      _max = 0;
    int      _idx = 0;
    obuf_fmt _fmt = { 10, 0, 0, ' '};

#if CC_DEBUG
    __GPU__ __INLINE__ void _debug(GT gt, U8 *v) {
        printf("%d>> obuf[%d] << ", blockIdx.x, _idx);
        switch(gt) {
        case GT_INT:   printf("%d\n", *(GI*)v);      break;
        case GT_FLOAT: printf("%G\n", *(GF*)v);      break;
        case GT_STR:   printf("%s\n", v);            break;
        case GT_FMT:   printf("%8x\n", *(U16*)v);    break;
        case GT_OPX: {
            OPX op = *v;
            U16 a  = (U16)*(v+1) | ((U16)*(v+2)<<8);
            U16 n  = (U16)*(v+3) | ((U16)*(v+4)<<8);
            switch (op) {
            case OPX_WORDS: printf("words()\n");            break;
            case OPX_SEE:   printf("see(%d)\n", a);         break;
            case OPX_DUMP:  printf("dump(%d, %d)\n", a, n); break;
            }
        } break;
        default:       printf("unknown type %d\n", gt);
        }
    }
    __GPU__ __INLINE__ void _dump() {
        for (int i=0; i<ALIGN16(_idx); i+=16) {
            printf("\n%04x: ", i);
            char w[17] = {0};
            for (int j=0; j<16; j++) {
                U8 c = _buf[i+j] & 0x7f;
                printf("%02x ", c);
                if (j%4==3) printf(" ");
                w[j] = (c==0x7f || c<0x20) ? '.' : c;
            }
            printf("%s", w);
        }
    }
#else  // CC_DEBUG
#define _debug(a,b)
#define _dump()
#endif // CC_DEBUG

    __GPU__  void _write(GT gt, U8 *v, int sz) {
        if (threadIdx.x!=0) return;               // only thread 0 within a block can write

        //_LOCK;
        obuf_node *n = (obuf_node*)&_buf[_idx];   // allocate next node

        n->gt   = gt;                             // data type
        n->id   = blockIdx.x;                     // VM.id
        n->sz   = ALIGN4(sz);                     // 32-bit alignment

        int inc = NODE_SZ + n->sz;                // calc node allocation size

        _debug(gt, v);

        if ((_idx + inc) > _max) inc = 0;         // overflow, skip
        else MEMCPY(n->data, v, sz);              // deep copy, TODO: shallow copy via managed memory

        _buf[(_idx += inc)] = (char)GT_EMPTY;     // advance index and mark end of stream
        //_UNLOCK;
        _dump();
    }
    __GPU__ Ostream& _wfmt() { _write(GT_FMT, (U8*)&_fmt, sizeof(obuf_fmt)); return *this; }

public:
    Ostream(int sz=CU4_OBUF_SZ) { cudaMallocManaged(&_buf, _max=sz); GPU_CHK(); }
    ~Ostream()                  { GPU_SYNC(); cudaFree(_buf); }
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
    __GPU__ Ostream& operator<<(_setbase b) { _fmt.base  = b.base;  return _wfmt(); }
    __GPU__ Ostream& operator<<(_setw    w) { _fmt.width = w.width; return _wfmt(); }
    __GPU__ Ostream& operator<<(_setprec p) { _fmt.prec  = p.prec;  return _wfmt(); }
    __GPU__ Ostream& operator<<(_setfill f) { _fmt.fill  = f.fill;  return _wfmt(); }
    ///
    /// object input
    ///
    __GPU__ Ostream& operator<<(char c) {
        char buf[2] = { c, '\0' };
        _write(GT_STR, (U8*)buf, 2);
        return *this;
    }
    __GPU__ Ostream& operator<<(GI i) {
        _write(GT_INT, (U8*)&i, sizeof(GI));
        return *this;
    }
    __GPU__ Ostream& operator<<(GF f) {
        _write(GT_FLOAT, (U8*)&f, sizeof(GF));
        return *this;
    }
    __GPU__ Ostream& operator<<(const char *s) {
        int len = STRLENB(s)+1;
        _write(GT_STR, (U8*)s, len);
        return *this;
    }
    __GPU__ Ostream& operator<<(_opx o) {
        U16 x[4] = { (U16)o.op, (U16)o.a, (U16)o.n, 0 };
        _write(GT_OPX, (U8*)x, sizeof(x));
        return *this;
    }
};
#endif // CUEF_SRC_OSTREAM_H_
