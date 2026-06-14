/** -*- c++ -*-
 * @file
 * @brief Cifar10 class - CIFAR-10 dataset provider host implemenation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <sstream>
#include <string>
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "cifar10.h"

namespace t4::ld {
///
/// for init debug MAX_BATCH 3
///
//#define MAX_BATCH 3          /**< debug, limit number of mini-batches */
#define MAX_BATCH 0          /**< debug, limit number of mini-batches */

Corpus *Cifar10::init(int mini_bsz, bool trace) {
    if (_open()) return NULL;
    data  = NULL;
    label = NULL;
    min   = 0;
    max   = 256;
    N     = mini_bsz;

    int n = 0;
    if (_ds) {
        _ds.seekg(0, std::ios::end);         /// * read to the end
        std::streampos fsz = _ds.tellg();    ///< file size
        _ds.seekg(0, std::ios::beg);         /// * rewind input file
        
        if (fsz % SAMPLE_BSZ) {
            ERROR("Cifar10::init file size %d != multiply of %d\n", (int)fsz, SAMPLE_BSZ);
            return 0;
        }
        n  = fsz / SAMPLE_BSZ;
        H  = IMAGE_H;
        W  = IMAGE_W;
        C  = IMAGE_C;
        
        if (trace) INFO("\tCIFAR-10 samples: [%d][%d,%d,%d]\n", n, H, W, C);
    }
    corpus_sz = n;
    batch_sz  = 0;

    return this;
}

int Cifar10::fetch(int bid, bool trace) {
    int off = N * bid;                          ///< batch offset index
    if (eof || off >= corpus_sz) {              /// * beyond total sample count?
        ERROR("Cifar10::fetch EOF reached (needs rewind)\n");
        eof=1; return 0;
    }
    ///
    /// fetch labels and images (and set eof if any of EOF reached)
    ///
    int n = batch_sz = _get_data(bid);          ///< load, update actual size loaded
    
    if ((off += n) >= corpus_sz) eof = 1;       /// * EOF reached
    ///
    /// control partial batch for debugging
    ///
    if (MAX_BATCH && ((bid+1) >= MAX_BATCH)) {  /// * forced stop? (debug)
        batch_sz = n >> 1;                      /// * fake a partial batch
        eof = 1;
    }
    if (trace) {
        INFO("\tCIFAR-10 batch[%d] loaded=%d/%d done=%d\n", bid, off, corpus_sz, eof);
    }
    return batch_sz;
}

int Cifar10::_open() {
    if (ds_name) {
        _ds.open(ds_name, std::ios::binary);
        if (!_ds.is_open()) { IO_ERROR(ds_name); return -1; }
    }
    return 0;
}

int Cifar10::_close() {
    if (_ds.is_open()) _ds.close();
    return 0;
}

int Cifar10::_get_data(int bid) {
    if (!label) DS_ALLOC(&label, N * LABEL_BSZ);   ///< allocate label memory block
    if (!data)  DS_ALLOC(&data,  N * IMAGE_BSZ);   ///< allocate image memory block

    int pos = bid * N * SAMPLE_BSZ;                ///< file pos
    _ds.seekg(pos);                                /// * seek by batch id

    std::vector<U8> buf(N * SAMPLE_BSZ);           ///< buffer on heap (CC: watch)
    _ds.read((char*)buf.data(), buf.size());       /// * read a mini-batch
    
    int cnt = (int)_ds.gcount();                   ///< bytes read this batch
    if (cnt % SAMPLE_BSZ) {
        ERROR("Cifar10::_get_data byte read %d != multiply of %d\n", cnt, SAMPLE_BSZ);
    }
    int n = cnt / SAMPLE_BSZ;                      ///< actual samples loaded
    
    U8 *bp = buf.data(), *tp = label, *dp = data;
    for (int i = 0; i < n; i++) {
        *tp++ = *bp;                               /// * read label
        bp += LABEL_BSZ;
        for (int j = 0; j < IMAGE_HW; j++) {       /// * image
            *dp++ = *(bp + j);                     /// * R
            *dp++ = *(bp + IMAGE_HW + j);          /// * G
            *dp++ = *(bp + IMAGE_HW * 2 + j);      /// * B
        }
        bp += IMAGE_BSZ;
    }
    char c = _ds.peek();
    eof |= _ds.eof();                              /// * set EOF flag, if done
    
    return n;                                      /// * number of samples fetched
}

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
