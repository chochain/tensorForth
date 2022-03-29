#ifndef CUEF_SRC_VECTOR_H
#define CUEF_SRC_VECTOR_H
#include "util.h"       /// also defined __GPU__

#define VECTOR_INC      4
///
/// Vector template class using device memory only
///
template<class T, int N=VECTOR_INC>
struct Vector {
    T   *v  = 0;         /// use proxy pattern
    int idx = 0;         /// number of elements stored
    int max = N;         /// allocated size

    __GPU__ Vector()               { v = new T[N]; }
    __GPU__ Vector(T a[], int len) { merge((T*)a, len); }
    __GPU__ Vector(Vector<T>& a)   { merge(a); }
    __GPU__ ~Vector()              { delete[] v; }
    //
    // operator overloading
    //
    __GPU__ T&      operator[](int i) { return i < 0 ? v[idx + i] : v[i]; }
    __GPU__ Vector& merge(T *a, int len) {
        for (int i=0; i<len; i++) push(a[i]);
        return *this;
    }
    __GPU__ Vector& merge(Vector<T>& a) {
        for (int i=0; i<a.idx; i++) push(a[i]);
        return *this;
    }
    __GPU__ Vector& operator+=(Vector<T>& a) { merge(a); return *this; }
    __GPU__ Vector& operator=(Vector<T>& a)  { idx=0; merge(a); return *this; }

    __GPU__ int     size() { return idx; }
    __GPU__ Vector& push(T t) {
        if ((idx+1) > max) resize(idx + VECTOR_INC);
        v[idx++] = t;
        return *this;
    }
    __GPU__ Vector& push(T *t, int sz) {
    	for (int i=0; i<sz; i++) push(*(t+i));
    }
    __GPU__ T&  pop()  { return idx>0 ? v[--idx] : v[0]; }
    __GPU__ T&  dec_i() { return v[idx - 1] -= 1; } /// decrement stack top
    __GPU__ Vector& clear(int i=0)  { if (i<idx) idx = i; return *this; }
    __GPU__ Vector& resize(int nsz) {
        int x = 0;
        if      (nsz >  max) x = ALIGN4(nsz);      // need bigger?
        else if (idx >= max) x = ALIGN4(idx + VECTOR_INC);  // allocate extra
        if (x==0) return *this;                    // no resizing needed
        // LOCK
        T *nv = new T[x];                          // allocate new block of memory
        if (v) {
            memcpy(nv, v, sizeof(T)*idx);          // deep copy
            delete[] v;
        }
        v   = nv;
        max = x;
        // UNLOCK
        return *this;
    }
};
#endif // CUEF_SRC_VECTOR_H

