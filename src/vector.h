#ifndef __EFORTH_SRC_VECTOR_H
#define __EFORTH_SRC_VECTOR_H
#include "util.h"       /// also defined __GPU__

#define VECTOR_INC          4
///
/// Vector template class
///
template<class T, int N=0>
struct Vector {
    T   *_v = 0;         /// use proxy pattern
    int _n  = 0;         /// number of elements stored
    int _sz = N;         /// allocated size

    __GPU__ Vector() { if (N) _v = new T[N]; }
    __GPU__ Vector(T a[], int len) { merge((T*)a, len); }
    __GPU__ Vector(Vector<T>& a)   { merge(a); }
    __GPU__ ~Vector() { if (_v) delete[] _v; }
    //
    // operator overloading
    //
    __GPU__ T&      operator[](int i) { return i < 0 ? _v[_n + i] : _v[i]; }
    __GPU__ Vector& merge(T *a, int len) {
        for (int i=0; i<len; i++) push(a[i]);
        return *this;
    }
    __GPU__ Vector& merge(Vector<T>& a) {
        for (int i=0; i<a._n; i++) push(a[i]);
        return *this;
    }
    __GPU__ Vector& operator+=(Vector<T>& a) { merge(a); return *this; }
    __GPU__ Vector& operator=(Vector<T>& a)  { _n=0; merge(a); return *this; }

    __GPU__ int     size() { return _n; }
    __GPU__ Vector& push(T t) {
        if ((_n+1) > _sz) resize(_n + VECTOR_INC);
        _v[_n++] = t;
        return *this;
    }
    __GPU__ T&  pop()  { return _n>0 ? _v[--_n] : _v[0]; }
    __GPU__ T&  dec_i() { return _v[_n - 1] -= 1; } /// decrement stack top
    __GPU__ Vector& clear(int i=0)  { if (i<_n) _n = i; return *this; }
    __GPU__ Vector& resize(int nsz) {
        int x = 0;
        if      (nsz >  _sz) x = ALIGN4(nsz);      // need bigger?
        else if (_n >= _sz)  x = ALIGN4(_n + VECTOR_INC);  // allocate extra
        if (x==0) return *this;                    // no resizing needed
        // LOCK
        T *nv = new T[x];                          // allocate new block of memory
        if (_v) {
            memcpy(nv, _v, sizeof(T)*_n);          // deep copy
            delete[] _v;
        }
        _v  = nv;
        _sz = x;
        // UNLOCK
        return *this;
    }
};
#endif // __EFORTH_SRC_VECTOR_H

