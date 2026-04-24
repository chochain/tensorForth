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

#if T4_DO_OBJ                /// * only when Object is activated

namespace t4 {
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
    DU  *data = NULL;        ///< managed memory block pointer (Note: instead of from TLSF)
    ///
    /// class contructors
    ///
    __HOST__ T4Base() :
        numel(0), rank(0) {}
    __HOST__ T4Base(U64 sz) :
        numel(sz), rank(1) {
        H_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U32 h, U32 w) :
        numel((U64)h * w), rank(2) {
        H_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U32 n, U32 h, U32 w, U32 c) :
        numel((U64)n * h * w * c), rank(4) {
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

} // namespace t4

#endif // T4_DO_OBJ
#endif // __T4BASE_H

