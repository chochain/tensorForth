/** -*- c++ -*-
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

__HOST__
Dataset::Dataset(U32 n, U32 h, U32 w, U32 c)
    : Tensor(n, h, w, c), label(NULL) {
    MM_ALLOC(&label, n * sizeof(U32));
    TRACE("Dataset[%d,%d,%d,%d] created\n", n, h, w, c);
}

__HOST__
Dataset::~Dataset() {
    if (!label) return;
    MM_FREE((void*)label);
}

__HOST__ void
Dataset::normalize(DU mean, DU scale) {
    _mean = mean;
    if (ZEQ(scale)) {
        ERROR("scale == 0?\n");
        _scale = 1.0f;
    }
    else _scale = 1.0f / scale;
}
//#define LOG_COUNT 300          /**< debug dump frequency */
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
///        -> corpus::rewind
///      -> corpus::fetch      - fetch host label/image blocks from files
///      -> _load              - transfer host blocks to device memory
/// Note:
///   ds_name: dataset name (match in loader.cu), for initial dataset setup
///   ds_name: NULL, following batch
///
__HOST__ int
    Dataset::fetch(char *ds_name, bool rewind, bool trace) {
    static const char *fn = "dataset#fetch";
    static long tick = 0;
    ///
    /// search cache for top <=> dataset pair
    ///
    if (trace) {
        INFO("  %s %s batch[%d] {\n",
             fn, ds_name ? ds_name : (rewind ? "rewind" : ""), batch_id);
    }
    Corpus *cp = ld::Loader::get(*this, ds_name); ///< Corpus/Dataset provider
    if (!cp) {
        ERROR("  } %s => not found in Loader\n", fn); return -1;
    }
    if (ds_name) {                                /// * init load
        if (cp->init(N(), trace)==NULL) {
            ERROR("  } %s => corpus init failed!\n", fn); return -2;
        }
        ///
        /// setsize is the total number of samples
        /// while N matches the sample size of mini-batch input tensor
        ///
        dataset_size = cp->corpus_sz;
        _reshape(cp->N, cp->H, cp->W, cp->C);     /// * reshape ds to match Corpus mini-batch
    }
    if (rewind) {
        cp->rewind();
        batch_id = done = 0;
    }
    ///
    /// load a mini-batch of data points
    ///
    if (!cp->fetch(batch_id, trace)) {            /// * fetch a batch from Corpus
        ERROR("  } %s => corpus fetch failed\n", fn);  return -3;
    }
    int n = batch_sz = cp->batch_sz;              ///< actural mini-batch fetched
    done = cp->eof;
    if (trace) {
        INFO("  } %s => batch[%d] ", fn, batch_id);
        if (done) INFO("completed, no more data.\n");
        else      INFO("%d record(s) loaded\n", n);
    } 
    ///
    /// transfer host into device memory
    /// if needed, allocate Dataset device (managed) memory blocks
    ///
    _load(cp->data, cp->label, n);                /// * transfer to device memory
    batch_id++;                                   /// * CC TODO: async prefetch
    ///
    /// debug tracing/preview
    ///
    if (LOG_COUNT > 1 && (++tick % LOG_COUNT)==0) {
        INFO("  batch[%d]/epoch, total batch = %ld\n", batch_id, tick);
        cp->show(n < 3 ? n : 3);
    }
    return 0;
}

__HOST__ void
Dataset::_load(U8 *cp_data, U8 *cp_label, int n) {
    int NX = n * HWC();
    ///
    /// Allocate managed memory if needed
    /// data and label buffer from Managed memory instead of TLSF
    /// Note: numel is known only after reading from Corpus
    ///       (see ~/src/io/aio_model#_dsfetch)
    ///
    if (!data)  {
        MM_ALLOC(&data,  sizeof(DU) * (numel+1));
        _tmp = &data[numel];                      /// * tmp storage for sum, std
    }
    if (!label) MM_ALLOC(&label, N() * sizeof(U32));
    ///
    /// scale cp_data into DU for nn/forward
    ///
    std::vector<DU> d(NX);                        ///< host buffer on heap (CC: watch)
    for (int i = 0; i < NX; i++, cp_data++) {     ///< NX < numel (partial mini-batch)
        d[i] = (I2D((int)*cp_data) - _mean) * _scale;  /// * normalize
    }
    H2D(data, d.data(), NX * sizeof(DU));         ///< data in managed memory
    ///
    /// scale cp_label into U32 for nn/loss
    ///
    U32 *t = (U32*)d.data();                      ///< reuse data buffer
    for (U32 i = 0; i < n; i++, cp_label++) {     ///< n < N (partial mini-batch)
        *t++ = (U32)*cp_label;                    /// * copy label to device memory
    }
    H2D(label, d.data(), n * sizeof(DU));         ///< label in managed memory
    
#if MM_DEBUG
    INFO("dataset.data=>");
    _dump(d.data(), H(), W(), C());
#endif // MM_DEBUG
}

} // namespace t4::mu

#endif  // (T4_DO_OBJ && T_DO_NN)
