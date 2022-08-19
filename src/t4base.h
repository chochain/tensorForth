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
    VIEW = 0,               ///< a view object
    TENSOR,                 ///< tensor object
    MODEL                   ///< NN model
} t4_obj;
///
/// tensorForth base object class
///
struct T4Base : public Managed {
    U32 numel;     ///< number of data elements, TODO: more than 4G elements
    union {
        U32  attr = 0;      ///< attrbutes collective
        struct {
            U8     dsize:5; ///< size of data element, F32 for now
            U8     rank :3; ///< rank of tensor 2:matrix, 4:NHWC tensor
            U8     parm;    ///< parameter storage
            U8     refc;    ///< reference counter (reserved)
            t4_obj ttype;   ///< 0: tensor, 1: view
        };
    };
    DU  *data;    ///< managed memory block pointer
    ///
    /// class contructors
    ///
    __HOST__ T4Base() :
        dsize(sizeof(DU)), numel(0), rank(0) {}
    __HOST__ T4Base(U32 sz) :
        dsize(sizeof(DU)), numel(sz), rank(1) {
        cudaMallocManaged((void**)&data, (size_t)numel * dsize);
        GPU_CHK();
    }
    __HOST__ T4Base(U16 h, U16 w) :
        dsize(sizeof(DU)), numel(h * w), rank(2) {
        cudaMallocManaged((void**)&data, (size_t)numel * dsize);
        GPU_CHK();
    }
    __HOST__ T4Base(U16 n, U16 h, U16 w, U16 c) :
        dsize(sizeof(DU)), numel(n * h * w * c), rank(4) {
        cudaMallocManaged((void**)&data, (size_t)numel * dsize);
        GPU_CHK();
    }
    __HOST__ ~T4Base() {
        if (!data) return;
        cudaFree((void*)data);
        switch (rank) {
        case 2: WARN("matrix(%d,%d) freed\n", shape[0], shape[1]); break;
        case 4: WARN("tensor(%d,%d,%d,%d) freed\n", shape[3], shape[0], shape[1], shape[2]); break;
        default: WARN("~Tensor error: rank=%d\n", rank);
        }
    }
    __BOTH__ __INLINE__ bool is_view()   { return ttype == VIEW;   }
    __BOTH__ __INLINE__ bool is_tensor() { return ttype <= TENSOR; }
    __BOTH__ __INLINE__ bool is_model()  { return ttype == MODEL;  }
};
#endif // TEN4_SRC_T4BASE_H_

