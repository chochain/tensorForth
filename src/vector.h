#ifndef __EFORTH_SRC_VECTOR_H
#define __EFORTH_SRC_VECTOR_H
#include "util.h"

namespace cuef {
///
/// vector template class
///
template<class T>
struct vector {
    T 	*v;             /// use proxy pattern
    int n  =0;          /// number of elements stored
    int sz =0;          /// allocated size

    __device__ ~vector() { if (v) free(v); }
    //
    // operator overloading
    //
    __device__ T&      operator[](int i) { return i < 0 ? v[n + i] : v[i]; }
    __device__ vector& operator+=(vector& a) {
    	for (int i=0; i<a.size(); i++) push(a[i]);
    	return *this;
    }
    __device__ vector& merge(T *a, int len) {
    	for (int i=0; i<len; i++) push(*a++);
    	return *this;
    }

    __device__ int     size() { return n; }
    __device__ vector& push(T t) {
        if ((n+1) > sz) resize(n + 4);
        v[n++] = t;
        return *this;
    }
    __device__ T& pop()   { return n>0 ? v[--n] : v[0]; }
    __device__ T  dec_i() { return v[n - 1] -= 1; }    /// decrement stack top
    __device__ vector& clear(int i=0)  { if (i<n) n = i; return *this; }
    __device__ vector& resize(int nsz) {
        int x = 0;
        if      (nsz > sz) x = ALIGN4(nsz);     // need bigger?
        else if (n >= sz)  x = ALIGN4(n + 4);	// auto allocate extra 4 elements
        if (x==0) return *this; 			    // no resizing needed

        T *nv = (T*)malloc(sizeof(T)*x);	    // allocate new block of memory
        MEMCPY(nv, v, sizeof(T)*sz);    	    // deep copy
        free(v);
        v  = nv;
        sz = x;
        return *this;
    }
};

} // namespace cuef
#endif // __EFORTH_SRC_VECTOR_H

