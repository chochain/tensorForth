/**
 * @file
 * @brief Dataset class - host-side dataset object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "dataset.h"

namespace t4::mu {

__HOST__ Dataset*
Dataset::load_batch(
    U8 *cp_data, U8 *cp_label, int n, DU mean, DU std) {
    const DU m = mean * 256, s = std * 256;
    const U64 nx = n * H() * W() * C();  ///< partial mini-batch
    ///
    /// Allocate managed memory if needed
    /// data and label buffer from Managed memory instead of TLSF
    /// Note: numel is known only after reading from Corpus
    ///       (see ~/src/io/aio_model#_dsfetch)
    ///
    if (!data)  MM_ALLOC(&data,  numel * sizeof(DU));
    if (!label) MM_ALLOC(&label, N() * sizeof(U32));
    ///
    /// scale cp_data into DU for nn/forward
    ///
    DU  *d = data;                       ///< data in managed memory
    for (U64 i = 0; i < numel; i++) {
        *d++ = (I2D((int)*cp_data) - m) / s;  /// * normalize
        if (i < nx) cp_data++;           /// * pad partial mini-batch
    }
    ///
    /// scale cp_label into U32 for nn/loss
    ///
    U32 *t = label;                      ///< label in managed memory
    for (U32 i = 0; i < N(); i++) {
        *t++ = (U32)*cp_label;           /// * copy label to device memory
        if (i < n) cp_label++;           /// * pad partial batch
    }
#if MM_DEBUG    
    INFO("dataset.data=>");
    _dump(data, H(), W(), C());
#endif // MM_DEBUG
        
    return this;
}

} // namespace t4::mu

#endif  // (T4_DO_OBJ && T_DO_NN)
