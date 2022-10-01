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
    int   batch_id = 0;             ///< current batch id
    DU    *label = NULL;            ///< label data on host
    ///
    /// constructors (for host testing mostly)
    ///
    __HOST__ Dataset(U16 n, U16 h, U16 w, U16 c)
        : Tensor(n, h, w, c) {
        batch_id = 0;
        MM_ALLOC((void**)&label, (size_t)n * sizeof(DU));
        WARN("Dataset[%d,%d,%d,%d] created\n", n, h, w, c);
    }
    __HOST__ ~Dataset() {
        if (!label) return;
        MM_FREE((void*)label);
    }
    __HOST__ Dataset &alloc(U16 n, U16 h, U16 w, U16 c) {
        WARN("Dataset::setup(%d, %d, %d, %d)\n", n, h, w, c);
        ///
        /// initialize
        ///
        numel    = (U32)n * h * w * c;
        dsize    = DSIZE;
        ttype    = T4_DATASET;
        batch_id = 0;
        reshape(n, h, w, c);
        ///
        /// allocate managed memory (not TLSF)
        ///
        MM_ALLOC(&data,  numel * sizeof(DU));
        MM_ALLOC(&label, N() * sizeof(DU));

        return *this;
    }
    __HOST__ Dataset *get_batch(U8 *h_data, U8 *h_label) {
        if (!data || !label) return NULL;
        DU  *d = data;
        for (int n = 0; n < numel; n++) {
            *d++ = I2D((int)*h_data++) / 256.0f;
        }
        DU  *t = label;
        for (int n = 0; n < N(); n++) {
            *t++ = I2D((int)*h_label++) / 256.0f;
        }
        return this;
    }
};
#endif  // TEN4_SRC_DATASET_H_

