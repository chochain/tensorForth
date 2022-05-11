/*! @file
  @brief
  cueForth Vector class (device only ) implementation

  TODO: Thrust::d_vector
*/
#ifndef CUEF_SRC_VECTOR_H
#define CUEF_SRC_VECTOR_H
#include "util.h"       /// also defined __GPU__

#define VECTOR_INC      4
///
/// Vector (device memory only) template class
///
template<typename T, int N=VECTOR_INC>
struct Vector {
    T   *v  = 0;         /// use proxy pattern
    int idx = 0;         /// number of elements stored
    int max = N;         /// allocated size

    __GPU__ Vector()             { v = new T[N]; }
    __GPU__ Vector(T a[], int n) { merge((T*)a, n); }
    __GPU__ Vector(Vector<T>& a) { merge(a); }
    __GPU__ ~Vector()            { delete[] v; }
    //
    // operator overloading
    //
    __GPU__ T&      operator[](int i) { return i < 0 ? v[idx + i] : v[i]; }
    __GPU__ Vector& push(T t)   {
        if ((idx+1) > max) resize(idx + VECTOR_INC);
        v[idx++] = t;                              /// deep copy
        return *this;
    }
    __GPU__ __INLINE__ Vector& operator<<(T t)    { push(t); }
    __GPU__ __INLINE__ Vector& merge(T *a, int n) {
        for (int i=0; i<n; i++) push(a[i]);
        return *this;
    }
    __GPU__ __INLINE__ Vector& merge(Vector<T>& a) {
        for (int i=0; i<a.idx; i++) push(a[i]);
        return *this;
    }
    __GPU__ __INLINE__ Vector& operator+=(Vector<T>& a) { merge(a); return *this; }
    __GPU__ __INLINE__ Vector& operator=(Vector<T>& a)  { idx=0; merge(a); return *this; }

    __GPU__ __INLINE__ Vector& push(T *t) { push(*t); }  /// aka copy constructor
    __GPU__ __INLINE__ Vector& push(T *t, int n) {
        for (int i=0; i<n; i++) push((t+i));
    }
    __GPU__ __INLINE__ int size()  { return idx; }
    __GPU__ __INLINE__ T&  pop()   { return idx>0 ? v[--idx] : v[0]; }
    __GPU__ __INLINE__ T&  dec_i() { return v[idx - 1] -= 1; } /// decrement stack top
    __GPU__ __INLINE__ Vector& clear(int i=0)  { if (i<idx) idx = i; return *this; }
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
