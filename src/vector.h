#ifndef __EFORTH_SRC_VECTOR_H
#define __EFORTH_SRC_VECTOR_H
#include "util.h"

#define ALIGN4(sz)          ((sz) + (-(sz) & 0x3))
#define ALIGN8(sz)          ((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)         ((sz) + (-(sz) & 0xf))

namespace cuef {
///
/// vector template class
///
template<class T>
struct vector {
    T   *_v;            /// use proxy pattern
    int _n  =0;         /// number of elements stored
    int _sz =0;         /// allocated size

    __GPU__ vector() {}
    __GPU__ vector(T a[], int len) { merge((T*)a, len); }
    __GPU__ ~vector() { if (_v) free(_v); }
    //
    // operator overloading
    //
    __GPU__ T&      operator[](int i) { return i < 0 ? _v[_n + i] : _v[i]; }
    __GPU__ vector& merge(T *a, int len) {
        for (int i=0; i<len; i++) push(a[i]);
        return *this;
    }
    __GPU__ vector& merge(vector<T>& a) {
        for (int i=0; i<a.size(); i++) push(a[i]);
        return *this;
    }
    __GPU__ vector& operator+=(vector<T>& a) { merge(a); return *this; }

    __GPU__ int     size() { return _n; }
    __GPU__ vector& push(T t) {
        if ((_n+1) > _sz) resize(_n + 4);
        _v[_n++] = t;
        return *this;
    }
    __GPU__ T&  pop()  { return _n>0 ? _v[--_n] : _v[0]; }
    __GPU__ T&  dec_i() { return _v[_n - 1] -= 1; } /// decrement stack top
    __GPU__ vector& clear(int i=0)  { if (i<_n) _n = i; return *this; }
    __GPU__ vector& resize(int nsz) {
        int x = 0;
        if      (nsz >  _sz) x = ALIGN4(nsz);      // need bigger?
        else if (_n >= _sz)  x = ALIGN4(_n + 4);   // auto allocate extra 4 elements
        if (x==0) return *this;                    // no resizing needed

        T *nv = (T*)malloc(sizeof(T)*x);           // allocate new block of memory
        if (_v) {
            memcpy(nv, _v, sizeof(T)*_n);          // deep copy
            free(_v);
        }
        _v  = nv;
        _sz = x;
        return *this;
    }
};

} // namespace cuef
#endif // __EFORTH_SRC_VECTOR_H

