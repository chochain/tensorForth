/**
 * @file
 * @brief Dataset class - host-side dataset object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __MMU_DATASET_H
#define __MMU_DATASET_H
#pragma once
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "tensor.h"

namespace t4::mu {

struct Dataset : public Tensor {
    int   setsize  =  0;             ///< total number of samples
    int   batch_id =  0;             ///< current batch id
    int   done     =  1;             ///< completed
    U32   *label;                    ///< label data on host
    ///
    /// constructors (for host testing mostly)
    ///
    __HOST__ Dataset(U32 n, U32 h, U32 w, U32 c)
        : Tensor(n, h, w, c), label(NULL) {
        MM_ALLOC(&label, n * sizeof(U32));
        TRACE("Dataset[%d,%d,%d,%d] created\n", n, h, w, c);
    }
    __HOST__ ~Dataset() {
        if (!label) return;
        MM_FREE((void*)label);
    }
    __HOST__ void normalize(DU mean, DU scale) {
        _mean = mean;
        if (ZEQ(scale)) {
            ERROR("scale == 0?\n");
            _scale = 1.0f;
        }
        else _scale = 1.0f / scale;
    }
    __HOST__ int fetch(char *ds_name, bool rewind, bool trace);

private:
    DU _mean  = 0.0f;
    DU _scale = 1.0f / 256.0f;
    
    __HOST__ void _reshape(U32 n, U32 h, U32 w, U32 c) {
        DEBUG("Dataset::setup(%d, %d, %d, %d)\n", n, h, w, c);
        ///
        /// set dimensions
        ///
        numel = (U64)n * h * w * c;    /// * number of batch elements
        Tensor::reshape(n, h, w, c);   /// * reshape to 4-D tensor
    }
    __HOST__ void _load(
        U8 *cp_data, U8 *cp_label, int n);
};

} // namespace t4::mu

#endif  // (T4_DO_OBJ && T_DO_NN)
#endif  // __MMU_DATASET_H

