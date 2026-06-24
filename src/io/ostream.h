/**
 * @file
 * @brief Ostream class - typed binary event queue for kernel-host marshalling
 *
 * Note: despite the name and << syntax, this is NOT a text stream.
 * It is a fixed flat-buffer binary event queue carrying typed payloads
 * (_opx, _tbx, DU, strings, ints) consumed by the host dispatcher.
 * std::ostream cannot replace it because of DU/IS_OBJ discrimination,
 * _opx/_tbx dispatch packets, and the binary obuf_fmt encoding.
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __IO_OSTREAM_H_
#define __IO_OSTREAM_H_
#pragma  once
#include "ten4_types.h"

namespace t4::io {
typedef std::ostream ostr;            ///< host output ostream
#define ENDL         '\n'
///
///> Binary event header — sits at every slot in _buf.
///  data() points to the byte immediately after the header.
///
struct event {
    U32 gt : 4;                       ///< GT type tag (16 types)
    U32 sz : 28;                      ///< payload size in bytes (max 256M)

    __HOST__ __INLINE__ U8* data() {
        return reinterpret_cast<U8*>(this) + sizeof(event);
    }
    __HOST__ __INLINE__ const U8* data() const {
        return reinterpret_cast<const U8*>(this) + sizeof(event);
    }
};

static_assert(sizeof(event) == sizeof(U32), "event header must be 4 bytes");

#define EVENT_HDR  sizeof(event)
///
///@name File Access Mode for IO Event
///@{
typedef enum {
    FAM_WO  = 0,
    FAM_RO  = 1,
    FAM_RW  = 2,
    FAM_RAW = 3
} FAM;
///@}
//================================================================
///
///> Binary format state — serialised as a GT_FMT payload, NOT text.
///  Cannot be replaced by std::setw etc. since those target text streams.
///
struct obuf_fmt {
    U8 base  = 10;
    U8 width = 0;
    U8 prec  = 0;
    U8 fill  = ' ';
    
    __HOST__ __INLINE__ void reset() { base=10; width=0; prec=0; fill=' '; }
};
///
///> Iomanip token structs — carry a single formatting field into operator<<.
///  Kept custom because they serialise into obuf_fmt binary payloads.
///
struct _setbase { U8 base;  __HOST__ _setbase(U8 b) : base(b)  {}};
struct _setw    { U8 width; __HOST__ _setw(U8 w)    : width(w) {}};
struct _setfill { U8 fill;  __HOST__ _setfill(U8 f) : fill(f)  {}};
struct _setprec { U8 prec;  __HOST__ _setprec(U8 p) : prec(p)  {}};

__HOST__ __INLINE__ _setbase setbase(int b)  { return _setbase((U8)b); }
__HOST__ __INLINE__ _setw    setw(int w)     { return _setw((U8)w);    }
__HOST__ __INLINE__ _setfill setfill(char f) { return _setfill((U8)f); }
__HOST__ __INLINE__ _setprec setprec(int p)  { return _setprec((U8)p); }
///
///> Forth operation dispatch packet.
///  op : 4  — operation enum (max 16 ops)
///  m  : 8  — mode / file access flags
///  i  : 20 — index argument (max 1M)
///  n       — DU float / tensor object id
///
struct _opx {
    U32 op : 4;   ///< complex object types (max 16 ops)
    U32 m  : 8;   ///< mode - file access, format
    U32 i  : 20;  ///< integer max 1M
    DU  n;        ///< F32 (tensor object id)
    
    __HOST__ _opx(OP op0, DU n0=DU0, U8 m0=0, int i0=0) : n(n0) {
        op = op0; m = m0; i = i0;
    }
};
///
///> TensorBoard dispatch packet.
///  op : 4  — TB_OP enum (max 16 ops)
///  i  : 28 — index argument (max 256M)
///  n       — DU float / tensor object id
///
struct _tbx {
    U32 op : 4;   ///< (TensorBoard ops) max 16 ops
    U32 i  : 28;  ///< max 256M
    DU  n;        ///< F32 (tensor object id)
    
    __HOST__ _tbx(TB_OP op0, DU n0=DU0, int i0=0) : n(n0) {
        op = op0; i = i0;
    }
};
///
///> Kernel-Host Convenience factory functions
///
__HOST__ __INLINE__ _opx opx(OP op, DU n=DU0, U8 m=0, int i=0) { return _opx(op, n, m, i); }
__HOST__ __INLINE__ _tbx tbx(TB_OP op, DU n=DU0, int i=0)      { return _tbx(op, n, i);    }
///
///> Ostream — typed binary event queue.
///
///  Layout of _buf:
///    [ event hdr (4B) | payload (ALIGN(sz) B) ] [ event hdr | payload ] ... GT_EMPTY
///
///  The host dispatcher walks _buf reading event headers and dispatching
///  by GT type. This is the only consumer of the binary format.
///
class Ostream : public OnHost {
    int      _max      = 0;
    int      _idx      = 0;
    bool     _overflow = false;      ///< set when a write was dropped
    obuf_fmt _fmt      = { 10, 0, 0, ' '};
    char    *_buf      = nullptr;

    __HOST__ void _debug(GT gt, U8 *d, U32 sz) {
#if T4_VERBOSE > 1
        printf("  ostr#_debug(gt=%x,sz=%d) obuf[%d] << ", gt, sz, _idx);
        if (!sz) return;

        switch(gt) {
        case GT_INT:   printf("%d",      *(IU*)d);  break;
        case GT_U32:   printf("%u",      *(U32*)d); break;
        case GT_FLOAT: printf("%G",      *(DU*)d);  break;
        case GT_STR:   printf("%s",      d);        break;
        case GT_OBJ:   printf("Obj[%lx]", (UFP)d);  break;
        case GT_FMT:   printf("%08x",    *(U32*)d); break;
        case GT_OPX: {
            _opx *o = (_opx*)d;
            switch (o->op) {
            case OP_DICT:  printf("dict_dump()");                   break;
            case OP_WORDS: printf("words()");                       break;
            case OP_SEE:   printf("see(%d)",      o->i);            break;
            case OP_DUMP:  printf("dump(%d, %d)", (U32)o->n, o->i); break;
            case OP_SS:    printf("ss_dump(%d)",  o->i);            break;
            case OP_DATA:  printf("data(%d)",     o->i);            break;
            case OP_NORM:  printf("norm(%d)",     o->i);            break;
            case OP_FETCH: printf("fetch(%d)",    o->i);            break;
            }
        } break;
        case GT_TBX: {
            _opx *o = (_opx*)d;
            switch (o->op) {
            case TB_INIT:  printf("tb_init()");    break;
            case TB_STEP:  printf("tb_step()");    break;
            case TB_SCALAR:printf("tb_num()");     break;
            case TB_TEXT:  printf("tb_text()");    break;
            case TB_IMAGE: printf("tb_image()");   break;
            case TB_TILE:  printf("tb_tile()");    break;
            case TB_HISTO: printf("tb_histo()");   break;
            case TB_GRAPH: printf("tb_graph()");   break;
            }
        } break;
        default: printf("unknown type %d", gt);
        }
        printf("\n");
#endif // T4_VERBOSE > 1
    }
    __HOST__  void _write(GT gt, U8 *vp, U32 sz) {
        //_LOCK;
        event* e = reinterpret_cast<event*>(&_buf[_idx]);  /// allocate next node

        e->gt = gt;                               /// data type
        e->sz = ALIGN(sz);                        /// data alignment (32-bit)

        int inc = (int)(EVENT_HDR + e->sz);       /// calc node allocation size

        if ((_idx + inc) > _max) {
            _overflow = true;                     /// * flag
            return;                               /// overflow, skip
        }
        memcpy(e->data(), vp, sz);                /// deep copy, TODO: shallow copy via managed memory
        _debug(gt, e->data(), e->sz);

        _buf[(_idx += inc)] = (char)GT_EMPTY;     /// advance index and mark end of stream
        //_UNLOCK;
    }
    __HOST__ Ostream& _wfmt() { _write(GT_FMT, (U8*)&_fmt, sizeof(obuf_fmt)); return *this; }

public:
    __HOST__ Ostream(U32 sz=T4_OBUF_SZ) { H_ALLOC(&_buf, _max=(int)sz);  }
    __HOST__ ~Ostream()                 { H_FREE(_buf); }
    ///
    /// clear output buffer
    ///
    __HOST__ Ostream& clear() {
        // LOCK
        _buf[_idx=0] = (char)GT_EMPTY;
        _overflow    = false;
        _fmt.reset();
        // UNLOCK
        return *this;
    }
    __HOST__ char *rdbuf()   { return _buf;      }
    __HOST__ U32  tellp()    { return (U32)_idx; }
    __HOST__ bool overflow() { return _overflow; }
    ///
    /// iomanip control
    ///
    __HOST__ Ostream& operator<<(_setbase b) { _fmt.base  = b.base;  return _wfmt(); }
    __HOST__ Ostream& operator<<(_setw    w) { _fmt.width = w.width; return _wfmt(); }
    __HOST__ Ostream& operator<<(_setprec p) { _fmt.prec  = p.prec;  return _wfmt(); }
    __HOST__ Ostream& operator<<(_setfill f) { _fmt.fill  = f.fill;  return _wfmt(); }
    ///
    /// object input
    ///
    __HOST__ Ostream& operator<<(char c) {
        char buf[4] = { c, '\0', '\0', '\0' };
        DEBUG("  ostr<<'%c'\n", c);
        _write(GT_STR, (U8*)buf, 4);
        return *this;
    }
    __HOST__ Ostream& operator<<(S32 i) {
        DEBUG("  ostr<<S32(%d)\n", i);
        _write(GT_INT, (U8*)&i, sizeof(S32));
        return *this;
    }
    __HOST__ Ostream& operator<<(U32 i) {
        DEBUG("  ostr<<U32(%u)\n", i);
        _write(GT_U32, (U8*)&i, sizeof(U32));
        return *this;
    }
    __HOST__ Ostream& operator<<(DU d) {
        GT t = IS_OBJ(d) ? GT_OBJ : GT_FLOAT;
        DEBUG("  ostr<<DU(gt=%d, %g)\n", t, d);
        _write(t, (U8*)&d, sizeof(DU));
        return *this;
    }
    __HOST__ Ostream& operator<<(const char *s) {
        DEBUG("  ostr<<\"%s\"\n", s);
        _write(GT_STR, (U8*)s, (U32)strlen(s)+1);
        return *this;
    }
    __HOST__ Ostream& operator<<(_opx x) {
        DEBUG("  ostr<<_opx\n");
        _write(GT_OPX, (U8*)&x, sizeof(x));
        return *this;
    }
    __HOST__ Ostream& operator<<(_tbx x) {
        DEBUG("  ostr<<_tbx\n");
        _write(GT_TBX, (U8*)&x, sizeof(x));
        return *this;
    }
};

} // namespace t4::io

#endif // __IO_OSTREAM_H_
