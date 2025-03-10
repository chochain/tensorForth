/**
 * @file
 * @brief Ostream class - kernel managed output stream module.
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __IO_OSTREAM_H_
#define __IO_OSTREAM_H_
#include "ten4_types.h"
#include "util.h"

//================================================================
/*! printf internal version data container.
*/
typedef struct {
    U8 base;
    U8 width;
    U8 prec;
    U8 fill;
} obuf_fmt;
///
/// implement kernel iomanip classes
///
struct _setbase { U8  base;  __GPU__ _setbase(U8 b) : base(b)  {}};
struct _setw    { U8  width; __GPU__ _setw(U8 w)    : width(w) {}};
struct _setfill { U8  fill;  __GPU__ _setfill(U8 f) : fill(f)  {}};
struct _setprec { U8  prec;  __GPU__ _setprec(U8 p) : prec(p)  {}};
__GPU__ __INLINE__ _setbase setbase(int b)  { return _setbase((U8)b); }
__GPU__ __INLINE__ _setw    setw(int w)     { return _setw((U8)w);    }
__GPU__ __INLINE__ _setfill setfill(char f) { return _setfill((U8)f); }
__GPU__ __INLINE__ _setprec setprec(int p)  { return _setprec((U8)p); }
///
/// Forth parameterized manipulators
///
struct _opx {
    union {
        U64 x;
        struct {
            U32 op : 4;   ///> max 16 ops
            U32 m  : 8;   ///> mode - file access, format
            U32 i  : 20;  ///> max 1M
            DU  n;        ///> F32
        };
    };
    __GPU__ _opx(OP op0, U8 m0, DU n, int i0=0) : n(n) { op = op0; m = m0; i = i0; }
};
__GPU__ __INLINE__ _opx opx(OP op, U8 m, DU n=DU0, int i=0) { return _opx(op, m, n, i); }
///
/// Ostream class
///
class Ostream : public Managed {
    char    *_buf;
    int      _max = 0;
    int      _idx = 0;
    obuf_fmt _fmt = { 10, 0, 0, ' '};

__GPU__ __INLINE__ void _debug(GT gt, U8 *v, U32 sz) {
#if T4_VERBOSE > 1
        printf("  ostr#_debug(gt=%x,sz=%d) obuf[%d] << ", gt, sz, _idx);
        if (!sz) return;
        U8 d[T4_STRBUF_SZ];
        MEMCPY(d, v, sz);
        switch(gt) {
        case GT_INT:   printf("%d",      *(IU*)d);  break;
        case GT_U32:   printf("%u",      *(U32*)d); break;
        case GT_FLOAT: printf("%G",      *(DU*)d);  break;
        case GT_STR:   printf("%s",      d);        break;
        case GT_OBJ:   printf("Obj[%x]", DU2X(d));  break;
        case GT_FMT:   printf("%08x",    *(U32*)d); break;
        case GT_OPX: {
            _opx *o = (_opx*)d;
            switch (o->op) {
            case OP_DICT:  printf("dict_dump()");                   break;
            case OP_WORDS: printf("words()");                       break;
            case OP_SEE:   printf("see(%d)",      o->i);            break;
            case OP_DUMP:  printf("dump(%d, %d)", o->i, (U32)o->n); break;
            case OP_SS:    printf("ss_dump(%d)",  o->i);            break;
            case OP_DATA:  printf("data(%d)",     o->i);            break;
            case OP_FETCH: printf("fetch(%d)",    o->i);            break;
            }
        } break;
        default: printf("unknown type %d", gt);
        }
        printf("\n");
#endif // T4_VERBOSE > 1
    }
    __GPU__  void _write(GT gt, U8 *v, U32 sz) {
        if (threadIdx.x!=0) return;               // only thread 0 within a block can write

        //_LOCK;
        io_event *e = (io_event*)&_buf[_idx];     // allocate next node

        e->gt   = gt;                             // data type
        e->sz   = ALIGN(sz);                      // data alignment (32-bit)

        int inc = EVENT_HDR + e->sz;              // calc node allocation size

        _debug(gt, v, sz);

        if ((_idx + inc) > _max) inc = 0;         // overflow, skip
        else MEMCPY(e->data, v, sz);              // deep copy, TODO: shallow copy via managed memory

        _buf[(_idx += inc)] = (char)GT_EMPTY;     // advance index and mark end of stream
        //_UNLOCK;
    }
    __GPU__ Ostream& _wfmt() { _write(GT_FMT, (U8*)&_fmt, sizeof(obuf_fmt)); return *this; }

public:
    Ostream(U32 sz=T4_OBUF_SZ) { MM_ALLOC(&_buf, _max=sz);  }
    ~Ostream()                 { GPU_SYNC(); MM_FREE(_buf); }
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
        DEBUG("  ostr#_write(char %c)\n", c);
        _write(GT_STR, (U8*)buf, 2);
        return *this;
    }
    __GPU__ Ostream& operator<<(S32 i) {
        DEBUG("  ostr#_write(S32) %d\n", i);
        _write(GT_INT, (U8*)&i, sizeof(S32));
        return *this;
    }
    __GPU__ Ostream& operator<<(U32 i) {
        DEBUG("  ostr#_write(U32) %d\n", i);
        _write(GT_U32, (U8*)&i, sizeof(U32));
        return *this;
    }
    __GPU__ Ostream& operator<<(DU d) {
        GT t = IS_OBJ(d) ? GT_OBJ : GT_FLOAT;
        DEBUG("  ostr#_write(DU) %d, %g\n", t, d);
        _write(t, (U8*)&d, sizeof(DU));
        return *this;
    }
    __GPU__ Ostream& operator<<(const char *s) {
        DEBUG("  ostr#_write(%s)\n", s);
        int len = STRLENB(s)+1;
        _write(GT_STR, (U8*)s, len);
        return *this;
    }
    __GPU__ Ostream& operator<<(_opx o) {
        DEBUG("  ostr#_write(_opx)\n");
        _write(GT_OPX, (U8*)&o, sizeof(o));
        return *this;
    }
};
#endif // __IO_OSTREAM_H_
