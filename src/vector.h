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

    ~vector() { if (v) free(v); }
    
    __GPU__ T&  operator[](int i) { return i < 0 ? v[n + i] : v[i]; }
    __GPU__ int size()    { return n; }
    __GPU__ vector *push(T t) {
        if ((n+1) > sz) resize(n + 4);
        v[n++] = t;
        return this;
    }
    __GPU__ T& pop()   { return n>0 ? v[--n] : NULL; }
    __GPU__ T  dec_i() { return v[n - 1] -= 1; }    /// decrement stack top
    __GPU__ void clear(int i=0)      { if (i<n) n = i; }
    __GPU__ void merge(vector& a) {
    	for (int i=0; i<a.size(); i++) push(a[i]);
    }
    __GPU__ void merge(T *a, int len) {
    	for (int i=0; i<len; i++) push(*a++);
    }
    __GPU__ void resize(int nsz) {
        U32 x = 0;
        if      (nsz > sz) x = ALIGN4(nsz);     	// need bigger?
        else if (n >= sz)  x = ALIGN4(n + 4);		// auto allocate extra 4 elements
        if (x==0) return;					    	// no resizing needed

        T *ndata = (T*)malloc(sizeof(T)*x);	    	// allocate new block of memory
        MEMCPY(ndata, data, sizeof(T)*sz);    	// deep copy
        free(data);                             
        data = ndata;
        sz = x;
    }
};

} // namespace cuef
#endif // __EFORTH_SRC_VECTOR_H

