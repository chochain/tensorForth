/**
 * @file
 * @brief Istream class - kernel input stream module
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_ISTREAM_H_
#define TEN4_SRC_ISTREAM_H_
#include "ten4_types.h"
#include "util.h"
#include "tensor.h"             /// in ../mmu include util.h
///
/// istream class
///
class Istream : public Managed {
    char *_buf;                 /// input buffer
    int  _idx  = 0;             /// current buffer index
    int  _gn   = 0;             /// number of byte processed

    __GPU__ int _tok(char delim) {
        char *p = &_buf[_idx];
        while (delim==' ' && (*p==' ' || *p=='\t')) (p++, _idx++); // skip leading blanks and tabs
        int nidx=_idx;
        while (*p && *p!=delim) (p++, nidx++);                     // advance pointers
        _gn = (delim!=' ' && *p!=delim) ? nidx=0 : nidx - _idx;    // not found or end of input string
        return nidx;
    }
public:
    Istream(int sz=T4_IBUF_SZ) { MM_ALLOC(&_buf, sz);       }
    ~Istream()                 { GPU_SYNC(); MM_FREE(_buf); }
    ///
    /// intialize by a given string
    ///
    __BOTH__ char     *rdbuf() { return _buf; }
    __BOTH__ Istream& clear()  {
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

        if (nidx > 0) {                     // token found
            MEMCPY(s, &_buf[_idx], _gn);    // CUDA memcpy
            _idx = nidx + (delim != ' ');   // advance index
            s[_gn] = '\0';                  // terminated with '\0'
            WARN("%d>> ibuf[%d] >> '%s' (%d bytes)\n", blockIdx.x, _idx, s, _gn);
        }
        else if (delim=='\n') {             // comment line
            WARN("%d>> ibuf[%d] \\ at idx=%d\n", blockIdx.x, _idx, _gn);
            _buf[_idx] = '\0';              // blank out the reset of input buffer
        }
        return *this;
    }
    __GPU__ Istream& getline(char *s, int sz, char delim='\n') {
        return get_idiom(s, delim);
    }
    __GPU__ int operator>>(char *s)   { get_idiom(s); return _gn; }
};
#endif // TEN4_SRC_ISTREAM_H_
