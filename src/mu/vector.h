/**
 * @file
 * @brief Vector class - device-side vector container module
 *
 * <pre>Copyright (C) 2021 GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 * Changelog:
 *   - Fixed: constructors called merge() with uninitialized v/idx/max (UB)
 *   - Fixed: destructor did not free v (memory leak)
 *   - Fixed: free() declared Vector& but had no return statement (UB)
 *   - Fixed: operator<<, push(T*), push(T*,int) missing return *this (UB)
 *   - Fixed: push() silently overflows buffer when VECTOR_DO_RESIZE=0
 *   - Fixed: pop() returned dangling reference to logically removed element
 *   - Fixed: init() did not reset idx, leaving stale count on reuse
 *   - Fixed: resize() used U32 param vs int fields (sign/type mismatch)
 *   - Fixed: copy ctor and array ctor did not initialize idx/max correctly
 *   - Added: const overloads for operator[] and size()
 *   - Added: bounds assertion in operator[] (debug mode)
 *   - Added: owned flag to distinguish external (init) vs owned (new) buffers
 */
#ifndef __MU_VECTOR_H
#define __MU_VECTOR_H
#include "ten4_types.h"
#include "util.h"            /// ALIGN, STRLEN, MEMCPY

#ifndef VECTOR_DO_RESIZE
#define VECTOR_DO_RESIZE  0
#endif
#define VECTOR_INC        4

namespace t4::mu {

template<typename T, int N=VECTOR_INC>
class Vector {
    T    *v     = NIL;      ///< data buffer (owned or external)
    int   idx   = 0;        ///< number of elements stored
    int   max   = 0;        ///< allocated capacity
    bool  owned = false;    ///< true if we allocated v ourselves
    
public:    
    ///
    /// Constructors / Destructor
    ///
    __HOST__ Vector() : idx(0), max(N), owned(true) {
        v = N ? new T[N] : NIL;
    }
    __HOST__ Vector(T a[], int n) : idx(0), max(n), owned(true) {
        v = n ? new T[n] : NIL;
        merge(a, n);
    }
    __HOST__ Vector(const Vector<T>& a) : idx(0), max(a.idx), owned(true) {
        v = a.idx ? new T[a.idx] : NIL;
        merge(const_cast<Vector<T>&>(a));
    }
    __HOST__ ~Vector() { _free(); }
    ///
    /// Buffer control
    ///
    /// Point this Vector at an external buffer (no ownership).
    /// Safe to call multiple times — releases any owned buffer first.
    __HOST__ __INLINE__ Vector& init(T *a, int n) {
        _free();
        v     = a;
        max   = n;
        idx   = 0;        ///< reset count; caller owns the buffer content
        owned = false;
        return *this;
    }
    /// Explicitly release owned buffer. No-op for external buffers.
    __HOST__ __INLINE__ void free() { _free(); }
    ///
    /// Element access
    ///
    __HOST__ __INLINE__ T& operator[](int i) {
        int ri = i < 0 ? idx + i : i;
        ASSERT(ri >= 0 && ri < idx);   ///< bounds check (debug)
        if (ri < 0 || ri >= idx) {
            printf("EEEEE vector.h i=%d, idx=%d => ri=%d\n", i, idx, ri);
        }
        return v[ri];
    }
    __HOST__ __INLINE__ const T& operator[](int i) const {
        int ri = i < 0 ? idx + i : i;
        ASSERT(ri >= 0 && ri < idx);
        return v[ri];
    }
    ///
    /// Stack / queue operations
    ///
    __HOST__ __INLINE__ Vector& push(T t) {
#if VECTOR_DO_RESIZE
        if ((idx + 1) > max) resize(idx + VECTOR_INC);
#else
        ASSERT(idx < max);             ///< catch overflow in debug
        if (idx >= max) return *this;  ///< silent guard in release
#endif
        v[idx++] = t;
        return *this;
    }
    /// Return by value — returning a reference to a popped slot is dangling
    __HOST__ __INLINE__ T pop() {
        ASSERT(idx > 0);
        return v[idx > 0 ? --idx : 0];
    }
    __HOST__ __INLINE__ Vector& clear(int i=0) {
        if (i >= 0 && i < idx) idx = i;
        return *this;
    }
    ///
    /// Merge / append
    ///
    __HOST__ __INLINE__ Vector& push(const T *t)         { return push(*t); }
    __HOST__ __INLINE__ Vector& push(const T *t, int n)  {
        for (int i = 0; i < n; i++) push(t[i]);
        return *this;
    }
    __HOST__ __INLINE__ Vector& merge(const T *a, int n) {
        for (int i = 0; i < n; i++) push(a[i]);
        return *this;
    }
    __HOST__ __INLINE__ Vector& merge(Vector<T>& a) {
        for (int i = 0; i < a.idx; i++) push(a.v[i]);
        return *this;
    }
    ///
    /// Operators
    ///
    __HOST__ __INLINE__ Vector& operator<<(T t)          { return push(t);  }
    __HOST__ __INLINE__ Vector& operator+=(Vector<T>& a) { return merge(a); }
    __HOST__ __INLINE__ Vector& operator=(const Vector<T>& a) {
        if (this == &a) return *this;
        idx = 0;
        /// grow owned buffer if needed
        if (owned && a.idx > max) {
            _free();
            v     = new T[a.idx];
            max   = a.idx;
            owned = true;
        }
        return merge(const_cast<Vector<T>&>(a));
    }
    ///
    /// Capacity
    ///
    __HOST__ __INLINE__ int  size()     const { return idx; }
    __HOST__ __INLINE__ int  capacity() const { return max; }
    __HOST__ __INLINE__ bool empty()    const { return idx == 0;   }
    __HOST__ __INLINE__ bool full()     const { return idx >= max; }
    ///
    /// Resize (only active when VECTOR_DO_RESIZE=1)
    ///
    __HOST__ Vector& resize(int nsz) {
#if VECTOR_DO_RESIZE
        ASSERT(owned);                          ///< cannot resize external buffer
        if (!owned) return *this;
        int x = 0;
        if      (nsz > max)  x = ALIGN(nsz);
        else if (idx >= max) x = ALIGN(idx + VECTOR_INC);
        if (x == 0) return *this;
        T *nv = new T[x];
        if (v) {
            memcpy(nv, v, sizeof(T) * idx);
            delete[] v;
        }
        v   = nv;
        max = x;
#endif
        return *this;
    }

private:
    __HOST__ __INLINE__ void _free() {
        if (owned && v) {
            delete[] v;
            v     = NIL;
            max   = 0;
            idx   = 0;
            owned = false;
        }
    }
};

} // namespace t4::mu

#endif // __MU_VECTOR_H
