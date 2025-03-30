/**
 * @file
 * @brief T4Base class - device-side base object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_types.h"

#if (!defined(__T4BASE_H) && T4_DO_OBJ)
#define __T4BASE_H
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
struct T4Base : public Managed {
    U64 numel;               ///< number of data elements
    union {
        U64 attr = 0;        ///< attrbutes collective
        struct {
            U32   ttype: 3;  ///< t4_obj, 0=tensor, 1=model, 2=dataset, 3=reserved
            U32   rank : 3;  ///< rank of tensor 2:matrix, 4:NHWC tensor
            U32   train: 1;  ///< trainable
            U32   dunit: 1;  ///< size of data element, F32=0, F64=1
            U32   nref : 8;  ///< reference counter (reserved)
            U32   iparm: 16; ///< integer parameter
            DU    xparm;      ///< float parameter
        };
    };
    DU  *data;    ///< managed memory block pointer (Note: instead of from TLSF)
    ///
    /// class contructors
    ///
    __HOST__ T4Base() :
        dunit(DUNIT), numel(0), rank(0) {}
    __HOST__ T4Base(U64 sz) :
        dunit(DUNIT), numel(sz), rank(1) {
        MM_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U32 h, U32 w) :
        dunit(DUNIT), numel((U64)h * w), rank(2) {
        MM_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ T4Base(U32 n, U32 h, U32 w, U32 c) :
        dunit(DUNIT), numel((U64)n * h * w * c), rank(4) {
        MM_ALLOC((void**)&data, (size_t)numel * sizeof(DU));
    }
    __HOST__ ~T4Base() {
        if (!data) return;
        MM_FREE((void*)data);
    }
    __BOTH__ __INLINE__ void init(U64 n, U8 tt, U8 rnk) {
        numel = n;
        ttype = tt;
        dunit = DUNIT;
        rank  = rnk;
        nref  = 1;
        iparm = 0;
        xparm = DU0;
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

#endif // (!defined(__T4BASE_H) && T4_DO_OBJ)

