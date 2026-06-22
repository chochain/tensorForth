/**
 * @file
 * @brief T4Base class - device-side base object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __T4BASE_H
#define __T4BASE_H
#pragma once
#include "ten4_types.h"

namespace t4 {

#if T4_DO_OBJ                /// * only when Object is activated
///
/// object classification macros
///
constexpr U32 T4_TYPE_MSK = 0x3;          ///< obj view flag
constexpr U32 T4_TT_OBJ   = 0x1;          ///< data unit flag
constexpr U32 T4_TT_VIEW  = 0x3;          ///< view of object
constexpr U32 EXT_FLAG    = 0x80000000;   /**< extention flag */

struct Variant {             /// * DU <=> pointer conversion utility class
    uintptr_t raw;
    Variant(void *ptr) : raw(reinterpret_cast<uintptr_t>(ptr)) {}
    U32  addr()        { return (U32)(raw & ~T4_TYPE_MSK); }
    bool is_obj()      { return (raw & T4_TT_OBJ) != 0; }
    bool is_view()     { return (raw & T4_TYPE_MSK)==T4_TT_VIEW; }
    void as_view()     { U32 *v = reinterpret_cast<U32*>(raw & ~T4_TYPE_MSK); *v |= T4_TT_VIEW; }
    void as_scalar()   { U32 *v = reinterpret_cast<U32*>(raw & ~T4_TYPE_MSK); *v &= ~T4_TT_OBJ; }
};
#define DU2X(v)     ((U32)Variant(&v).raw)           /**< to U32 ptr     */
#define SCALAR(v)   (Variant(&v).as_scalar())        /**< set DU flag    */

#define IS_OBJ(v)   (Variant(&v).is_obj())           /**< if is an obj   */
#define IS_VIEW(v)  (Variant(&v).is_view())
#define AS_VIEW(v)  (Variant(&v).as_view(), (v))
///
/// tensorForth object types
///
typedef enum {
    T4_TENSOR = 0,           ///< tensor object
    T4_MODEL,                ///< NN model
    T4_DATASET,              ///< NN dataset
    T4_XXX                   ///< reserved
} t4_obj;
///
/// tensorForth base object class
///
struct T4Base : public OnHost {
    U64 numel;               ///< number of data elements
    union {
        U64 attr = 0;        ///< attrbutes collective
        struct {
            U32   ttype: 3;  ///< t4_obj, 0=tensor, 1=model, 2=dataset, 3=reserved
            U32   rank : 3;  ///< rank of tensor 2:matrix, 4:NHWC tensor
            U32   iparm: 10; ///< integer parameter or method id
            U32   nref : 13; ///< reference counter (reserved)
            U32   train: 1;  ///< trainable
            U32   f64  : 1;  ///< size of data element, F32=0, F64=1
            U32   err  : 1;  ///< math error (NaN, Inf)
            DU    xparm;     ///< float parameter
        };
    };
    DU  *data;               ///< managed memory block pointer (Note: instead of from TLSF)
    ///
    /// class contructors
    ///
    __HOST__ T4Base() :
        numel(0), rank(0), data(NULL) {}
    __HOST__ T4Base(U64 sz) :
        numel(sz), rank(1) {
        H_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U32 h, U32 w) :
        numel((U64)h * w), rank(2) {
        H_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U32 n, U32 h, U32 w, U32 c) :
        numel((U64)n * h * w * c), rank(4), data(NULL) {
        H_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ ~T4Base() {
        if (!data) return;
        H_FREE((void*)data);
    }
    __HOST__ __INLINE__ void init(U64 n, U8 tt, U8 rnk) {
        numel = n;
        ttype = tt;
        rank  = rnk;
        train = 0;
        err   = 0;
        nref  = 1;
        iparm = 0;
        xparm = DU0;
        data  = NULL;
    }
    __HOST__ __INLINE__ DU   &operator[](int i) { return data[i]; }
    __HOST__ __INLINE__ int  ref_inc() {
        int r = ++nref;                     /// TODO: atomicAdd
//        printf("nref=%d\n", r);
        return r;
    }
    __HOST__ __INLINE__ int  ref_dec() {
        if (nref > 1) {
            int r = --nref;                 /// TODO: atomicSub
//            printf("nref=%d\n", r);
            return r;
        }
        return 0;
    }
    __HOST__ __INLINE__ bool is_tensor()  { return ttype == T4_TENSOR;  }
    __HOST__ __INLINE__ bool is_model()   { return ttype == T4_MODEL;   }
    __HOST__ __INLINE__ bool is_dataset() { return ttype == T4_DATASET; }
};

#ifdef __CUDACC__     // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
///
///@name CUDA kernel launching macros
///@note: consider use of fn<<<_g,_b,0,cudaStreamTailLaunch>>>(...)
///@{
#define WARP_SUM(v)                                         \
    for (int off = 16; off > 0; off >>=1)                   \
        v += __shfl_down_sync(0xffffffff, v, off)
#define WARP_MAX(v)                                         \
    for (int off = 16; off > 0; off >>= 1)                  \
        v = MAX(v, __shfl_down_sync(0xffffffff, v, off))

#define FORK(fn,n,...) {                                    \
    const dim3 _b(T4_DIM_SQ, 1, 1);                         \
    const dim3 _g(((n) + _b.x - 1) / _b.x, 1, 1);           \
    fn<<<_g,_b>>>(__VA_ARGS__,n);                           \
    GPU_CHK();                                              \
}
#define FORK1(fn, c, n, ...) {                              \
    const dim3 _g((c), (n), 1);                             \
    fn<<<_g,T4_DIM_SZ2>>>(__VA_ARGS__);                     \
    GPU_CHK();                                              \
}
#define FORK2(fn,_g,n,...) {                                \
    fn<<<_g,T4_DIM_SQ>>>(__VA_ARGS__,n);                    \
    GPU_CHK();                                              \
}
#define FORK3(fn,h,w,c,...) {                               \
    const dim3 _b(T4_DIM_SZ, T4_DIM_SZ, 1);                 \
    const dim3 _g(((w) + _b.x - 1) / _b.x,                  \
                  ((h) + _b.y - 1) / _b.y, c);              \
    fn<<<_g,_b>>>(__VA_ARGS__,h,w);                         \
    GPU_CHK();                                              \
}
#define FORK4(fn,sm,...) { /** N,H,W,C (default params) */  \
    const dim3 _b(T4_DIM_SQ, 1, 1);                         \
    const dim3 _g(((W)*(H) + _b.x - 1) / _b.x, C, N);       \
    fn<<<_g,_b,sm>>>(__VA_ARGS__);                          \
    GPU_CHK();                                              \
}
///@}
struct Managed {
    void *operator new(size_t sz) {
        void *ptr;
        MM_ALLOC(&ptr, sz);
        DEBUG("new Managed Obj %p size=%ld byes\n", ptr, sz);
        return ptr;
    }
    void operator delete(void *ptr) { MM_FREE(ptr); }
};
#endif // __CUDACC__  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#else  // !T4_DO_OBJ
#define IS_OBJ(v)   (0)
#define IS_VIEW(v)  (0)
#define AS_VIEW(v)

#endif // T4_DO_OBJ

} // namespace t4
#endif // __T4BASE_H

