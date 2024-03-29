/**
 * @file
 * @brief T4Base class - device-side base object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_T4BASE_H
#define TEN4_SRC_T4BASE_H
#include "ten4_types.h"

#if T4_ENABLE_OBJ
///
/// tensorForth object types
///
typedef enum {
    T4_TENSOR = 0,          ///< tensor object
    T4_MODEL,               ///< NN model
    T4_DATASET,             ///< NN dataset
    T4_XXX                  ///< reserved
} t4_obj;
///
/// tensorForth base object class
///
struct T4Base : public Managed {
    U32 numel;     ///< number of data elements, TODO: more than 4G elements
    union {
        U32  attr = 0;      ///< attrbutes collective
        struct {
            U8    ttype: 2; ///< t4_obj, 0=tensor, 1=model, 2=dataset, 3=reserved
            U8    dunit: 2; ///< size of data element, F32=0, F64=1
            U8    rank : 3; ///< rank of tensor 2:matrix, 4:NHWC tensor
            U8    train: 1; ///< trainable
            U8    nref;     ///< reference counter (reserved)
            S16   parm;     ///< extra parameter storage
        };
    };
    DU  *data;    ///< managed memory block pointer (Note: instead of from TLSF)
    ///
    /// class contructors
    ///
    __HOST__ T4Base() :
        dunit(DUNIT), numel(0), rank(0) {}
    __HOST__ T4Base(U32 sz) :
        dunit(DUNIT), numel(sz), rank(1) {
        MM_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U16 h, U16 w) :
        dunit(DUNIT), numel(h * w), rank(2) {
        MM_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U16 n, U16 h, U16 w, U16 c) :
        dunit(DUNIT), numel(n * h * w * c), rank(4) {
        MM_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ ~T4Base() {
        if (!data) return;
        MM_FREE((void*)data);
    }
    __BOTH__ __INLINE__ void init(U32 n, U8 tt, U8 rnk) {
        numel = n;
        ttype = tt;
        dunit = DUNIT;
        rank  = rnk;
        nref  = 1;
        parm  = 0;
        data  = NULL;
    }
    __BOTH__ __INLINE__ DU   &operator[](int i) { return data[i]; }
    __BOTH__ __INLINE__ int  ref_inc() {
        int r = ++nref;                     /// TODO: atomicAdd
//        printf("nref=%d\n", r);
        return r;
    }
    __BOTH__ __INLINE__ int  ref_dec() {
        if (nref > 1) {
            int r = --nref;                 /// TODO: atomicSub
//            printf("nref=%d\n", r);
            return r;
        }
        return 0;
    }
    __BOTH__ __INLINE__ bool is_tensor()  { return ttype == T4_TENSOR;  }
    __BOTH__ __INLINE__ bool is_model()   { return ttype == T4_MODEL;   }
    __BOTH__ __INLINE__ bool is_dataset() { return ttype == T4_DATASET; }
};

#endif // T4_ENABLE_OBJ
#endif // TEN4_SRC_T4BASE_H

