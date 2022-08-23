/**
 * @file
 * @brief tensorForth - base object class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_T4BASE_H_
#define TEN4_SRC_T4BASE_H_
#include "ten4_types.h"
///
/// tensorForth object types
///
typedef enum {
    T4_VIEW = 0,            ///< a view object
    T4_TENSOR,              ///< tensor object
    T4_MODEL                ///< NN model
} t4_obj;
///
/// tensorForth base object class
///
struct T4Base : public Managed {
    U32 numel;     ///< number of data elements, TODO: more than 4G elements
    union {
        U32  attr = 0;      ///< attrbutes collective
        struct {
            U8    ttype: 2; ///< t4_obj, 0:view, 1=tensor, 2=model
            U8    dsize: 3; ///< size of data element, F32=5, F64=6
            U8    rank : 3; ///< rank of tensor 2:matrix, 4:NHWC tensor
            U8    nref;     ///< reference counter (reserved)
            U16   parm;     ///< extra parameter storage
        };
    };
    DU  *data;    ///< managed memory block pointer
    ///
    /// class contructors
    ///
    __HOST__ T4Base() :
        dsize(DSIZE), numel(0), rank(0) {}
    __HOST__ T4Base(U32 sz) :
        dsize(DSIZE), numel(sz), rank(1) {
        cudaMallocManaged((void**)&data, (size_t)numel * dsize);
        GPU_CHK();
    }
    __HOST__ T4Base(U16 h, U16 w) :
        dsize(DSIZE), numel(h * w), rank(2) {
        cudaMallocManaged((void**)&data, (size_t)numel * dsize);
        GPU_CHK();
    }
    __HOST__ T4Base(U16 n, U16 h, U16 w, U16 c) :
        dsize(DSIZE), numel(n * h * w * c), rank(4) {
        cudaMallocManaged((void**)&data, (size_t)numel * dsize);
        GPU_CHK();
    }
    __HOST__ ~T4Base() {
        if (!data) return;
        cudaFree((void*)data);
    }
    __BOTH__ __INLINE__ bool is_view()   { return ttype == T4_VIEW;   }
    __BOTH__ __INLINE__ bool is_tensor() { return ttype <= T4_TENSOR; }
    __BOTH__ __INLINE__ bool is_model()  { return ttype == T4_MODEL;  }
};
#endif // TEN4_SRC_T4BASE_H_

