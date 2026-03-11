/**
 * @file
 * @brief Dataset class - host-side dataset object
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "ld/loader.h"   /// includes Corpus
#include "dataset.h"

namespace t4::mu {
using ld::Corpus;
///
/// for init debug LOG_COUNT 1 with sample=3 is good
///
#define LOG_COUNT 1          /**< debug dump frequency */
///
/// initial dataset setup
/// init flow:
///    netvm::dataset
///    -> sys::process_event   OP_DATA
///    -> mmu::dataset         - set N=batch_sz, batch_id = -1
///
/// fetch flow:
///    netvm::fetch
///    -> sys::process_event   OP_FETCH
///    -> dataset::fetch
///      if (ds_name != null)  - init
///        -> corpus::init     - setup dimensions
///        -> _reshape         - set dimensions for the first batch (no alloc yet)
///      -> corpus::fetch      - fetch host label/image blocks from files
///      -> _load              - transfer host blocks to device memory
/// Note:
///   ds_name: dataset name (match in loader.cu), for initial dataset setup
///   ds_name: NULL, following batch
///
__HOST__ int
Dataset::fetch(char *ds_name, bool rewind) {
    static const char *fn = "dataset#dsfetch";
    static int tick = 0;
    ///
    /// search cache for top <=> dataset pair
    ///
    TRACE("  %s(%s) dataset (batch_id=%d) {\n",
          fn, ds_name ? ds_name : (rewind ? "rewind" : "fetch"), batch_id);
    Corpus *cp = ld::Loader::get(*this, ds_name); ///< Corpus/Dataset provider
    if (!cp) {
        ERROR("  } %s => dataset not found\n", fn); return -1;
    }
    if (ds_name) {                                /// * init load
        if (cp->init()==NULL) {
            ERROR("  } %s => dataset setup failed!\n", fn); return -2;
        }
        _reshape(N(), cp->H, cp->W, cp->C);       /// * reshape ds to match Corpus mini-batch
    }
    if (rewind) {
        cp->rewind();
        batch_id = done = 0;
    }
    else if ((done=cp->eof)) {                    /// * dataset exhausted?
        TRACE("  } %s => completed, no more data.\n", fn); return 0;
    }
    ///
    /// load a mini-batch of data points
    ///
    if (!cp->fetch(batch_id, N())) {              /// * fetch a batch from Corpus
        ERROR("  } %s => fetch failed\n", fn);  return -3;
    }
    int n = cp->batch_sz;                         ///< actural mini-batch fetched
    ///
    /// transfer host into device memory
    /// if needed, allocate Dataset device (managed) memory blocks
    ///
    _load(cp->data, cp->label, n);
    TRACE("  } %s => batch[%d] %d record(s) loaded, done=%d\n",
          fn, batch_id, n, cp->eof);

    batch_id++;
    done = cp->eof;
    
    if (LOG_COUNT && ++tick == LOG_COUNT) {
        cp->show(n < 3 ? n : 3);
        tick = 0;
    }
    return 0;
}

__HOST__ void
Dataset::_load(U8 *cp_data, U8 *cp_label, int n, DU mean, DU std) {
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
    DU  *d = data;                            ///< data in managed memory
    for (U64 i = 0; i < nx; i++, cp_data++) { ///< nx < numel (partial mini-batch)
        *d++ = (I2D((int)*cp_data) - m) / s;  /// * normalize
    }
    ///
    /// scale cp_label into U32 for nn/loss
    ///
    U32 *t = label;                           ///< label in managed memory
    for (U32 i = 0; i < n; i++, cp_label++) { ///< n < N (partial mini-batch)
        *t++ = (U32)*cp_label;                /// * copy label to device memory
    }
#if MM_DEBUG    
    INFO("dataset.data=>");
    _dump(data, H(), W(), C());
#endif // MM_DEBUG
}

} // namespace t4::mu

#endif  // (T4_DO_OBJ && T_DO_NN)
