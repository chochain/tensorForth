/**
 * @file
 * @brief Dataset class - host-side dataset object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#pragma once
#include "ten4_config.h"

#if (!defined(__MMU_DATASET_H) && T4_DO_OBJ && T4_DO_NN)
#define __MMU_DATASET_H
#include "tensor.h"

namespace t4::mu {

struct Dataset : public Tensor {
    int   batch_id =  0;             ///< current batch id
    int   done     =  1;             ///< completed
    U32   *label   = NULL;           ///< label data on host
    ///
    /// constructors (for host testing mostly)
    ///
    __HOST__ Dataset(U32 n, U32 h, U32 w, U32 c)
        : Tensor(n, h, w, c) {
        MM_ALLOC(&label, n * sizeof(U32));
        TRACE("Dataset[%d,%d,%d,%d] created\n", n, h, w, c);
    }
    __HOST__ ~Dataset() {
        if (!label) return;
        MM_FREE((void*)label);
    }
    __HOST__ Dataset &reshape(U32 n, U32 h, U32 w, U32 c) {
        DEBUG("Dataset::setup(%d, %d, %d, %d)\n", n, h, w, c);
        ///
        /// set dimensions
        ///
        numel = (U64)n * h * w * c;    /// * number of batch elements
        Tensor::reshape(n, h, w, c);   /// * reshape to 4-D tensor
        
        return *this;
    }
    __HOST__ Dataset *load_batch(
        U8 *cp_data, U8 *cp_label, int n, DU mean=DU0, DU std=DU1);
};

} // namespace t4::mu

#endif  // (!defined(__MMU_DATASET_H) && T4_DO_OBJ && T_DO_NN)

