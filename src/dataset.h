/**
 * @file
 * @brief tensorForth Dataset class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_DATASET_H_
#define TEN4_SRC_DATASET_H_
#include "t4base.h"
#include "ndata.h"

struct Dataset : public T4Base {
    U8    *label = NULL;      ///< label data on host
    
    U16   N, H, W, C;         ///< dimensions
    U16   batch_id = 0;       ///< current batch id
    ///
    /// constructors (for host testing mostly)
    ///
    __HOST__ Dataset(U16 n, U16 h, U16 w, U16 c)
        : T4Base(n, h, w, c) {
        N = n; H = h; W = w; C = c;
        batch_id = 0;
        MM_ALLOC((void**)&label, (size_t)n * sizeof(DU));
        WARN("Dataset[%d,%d,%d,%d] created\n", n, h, w, c);
    }
    __HOST__ ~Dataset() {
        if (!label) return;
        MM_FREE((void*)label);
    }
    __HOST__ Dataset &setup(U16 h, U16 w, U16 c) {
        WARN("Dataset::setup(%d, %d, %d)\n", h, w, c);
        H = h; W = w; C = c;
        batch_id = 0;

        numel = (U32)N * H * W * C;    /// * T4Base members
        attr  = (4 << 5) | (DSIZE << 2) | T4_DATASET;
        
        MM_ALLOC((void**)&data,  numel * sizeof(DU));
        MM_ALLOC((void**)&label, N * sizeof(DU));
        
        return *this;
    }
    __BOTH__ int     dsize() { return H * W * C; }
    __BOTH__ int     len()   { return N; }
    ///
    /// IO
    ///
    virtual Dataset *get_batch() {
        /*
        int   bsz = batch_sz * dsize() * sizeof(DU);
        float *d  = (DU*)data;
        U8    *s  = (U8*)nd->data;
        for (int n = 0; n < bsz; n++) {
            *d++ = static_cast<float>(*s) / 256.0f;
        }
        */
        return this;
    }
};
#endif  // TEN4_SRC_DATASET_H_

