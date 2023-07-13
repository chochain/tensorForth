/**
 * @file
 * @brief Dataset class - host-side basic data object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_DATASET_H_
#define TEN4_SRC_DATASET_H_
#include "tensor.h"                  // in ../mmu

struct Dataset : public Tensor {
    int   batch_id = -1;            ///< current batch id
    int   done     = -1;            ///< completed
    U16   *label = NULL;            ///< label data on host
    ///
    /// constructors (for host testing mostly)
    ///
    __HOST__ Dataset(U16 n, U16 h, U16 w, U16 c)
        : Tensor(n, h, w, c) {
        MM_ALLOC(&label, n * sizeof(U16));
        batch_id = 0;
        WARN("Dataset[%d,%d,%d,%d] created\n", n, h, w, c);
    }
    __HOST__ ~Dataset() {
        if (!label) return;
        MM_FREE((void*)label);
    }
    __HOST__ Dataset &reshape(U16 n, U16 h, U16 w, U16 c) {
        WARN("Dataset::setup(%d, %d, %d, %d)\n", n, h, w, c);
        ///
        /// set dimensions
        ///
        numel = (U32)n * h * w * c;    /// * number of batch elements
        Tensor::reshape(n, h, w, c);   /// * reshape to 4-D tensor
        
        batch_id = 0;                  /// * signify batch dimension set now
        
        return *this;
    }
    __HOST__ Dataset *load_batch(U8 *h_data, U8 *h_label) {
        ///
        /// Note: from Managed memory instead of TLSF
        /// allocate managed memory if needed
        /// it's necessary because numel is known only after reading from Corpus
        /// (see ~/src/io/aio_model#_dsfetch)
        ///
        if (!data)  MM_ALLOC(&data, numel * sizeof(DU));
        if (!label) MM_ALLOC(&label, N() * sizeof(U16));
        
        DU  *d = data;
        for (int i = 0; i < numel; i++) {
            *d++ = I2D((int)*h_data++) / 256.0f;
        }
        U16 *t = label;
        for (int i = 0; i < N(); i++) {
            *t++ = (U16)*h_label++;
        }
        return this;
    }
};
#endif  // TEN4_SRC_DATASET_H_

