/**
 * @file
 * @brief Dataset class - host-side basic data object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#if (!defined(__MMU_DATASET_H) && T4_DO_OBJ && T4_DO_NN)
#define __MMU_DATASET_H
#include "mmu/tensor.h"

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
        TRACE("Dataset::setup(%d, %d, %d, %d)\n", n, h, w, c);
        ///
        /// set dimensions
        ///
        numel = (U64)n * h * w * c;    /// * number of batch elements
        Tensor::reshape(n, h, w, c);   /// * reshape to 4-D tensor
        
        return *this;
    }
    __HOST__ Dataset *load_batch(
        U8 *h_data, U8 *h_label, DU mean=DU0, DU std=DU1) {
        const DU m = mean * 256, s = std * 256;
        ///
        /// Allocate managed memory if needed
        /// data and label buffer from Managed memory instead of TLSF
        /// Note: numel is known only after reading from Corpus
        ///       (see ~/src/io/aio_model#_dsfetch)
        ///
        if (!data)  MM_ALLOC(&data, numel * sizeof(DU));
        if (!label) MM_ALLOC(&label, N() * sizeof(U32));

        DU  *d = data;                ///< data in device memory
        for (U64 i = 0; i < numel; i++) {
            *d++ = (I2D((int)*h_data++) - m) / s;  // normalize
        }
        U32 *t = label;               ///< label in device memory
        for (U64 i = 0; i < N(); i++) {
            *t++ = (U32)*h_label++;
        }
        return this;
    }
};

#endif  // (!defined(__MMU_DATASET_H) && T4_DO_OBJ && T4_DO_NN)

