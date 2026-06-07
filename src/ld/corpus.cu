/** -*- c++ -*-
 * @file
 * @brief Corpus base class - dataset provider host implemenation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <sstream>
#include <string>
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "corpus.h"

namespace t4::ld {
///
/// for init debug MAX_BATCH 3
///
//#define MAX_BATCH 3          /**< debug, limit number of mini-batches */
#define MAX_BATCH 0          /**< debug, limit number of mini-batches */

Corpus *Corpus::show(int N) {
    static const char *map = " .:-=+*#%@";

    for (int i = 0; i < H; i++) {
        for (int n =0; n < N; n++) {
            U8 *img = (*this)[n] + i * W;
            for (int j = 0; j < W; j++, img++) {
                char c  = map[*img / 26];
                char c1 = map[((int)*img + (int)*(img+1)) / 52];
                INFO("%c%c", c, c1);                 /// double width
            }
            INFO("|");
        }
        INFO("\n");
    }
    for (int n = 0; n < N; n++) {
        INFO(" label=%-2d ", (int)label[n]);
        for (int j = 0; j < W*2 - 10; j++) INFO("-");
        INFO("+");
    }
    INFO("\n");

    return this;
}
///
/// tracing to make sure process is going
///
int Corpus::_open() {
    if (ds_name) {
        _ds.open(ds_name, std::ios::binary);
        if (!_ds.is_open()) { IO_ERROR(ds_name); return -1; }
    }
    if (tg_name) {
        _tg.open(tg_name, std::ios::binary);
        if (!_tg.is_open()) { IO_ERROR(tg_name); return -1; }
    }
    return 0;
}

int Corpus::_close() {
    if (_ds.is_open()) _ds.close();
    if (_tg.is_open()) _tg.close();
    return 0;
}

int Corpus::_get_labels(int bid, int hdr, int bsz) {
    if (!label) DS_ALLOC(&label, bsz);             ///< allocate managed memory

    _tg.seekg(hdr + bid * bsz);                    /// * seek by batch
    _tg.read((char*)label, bsz);                   /// * fetch batch labels

    int cnt = _tg.gcount();                        ///< # of labels extracted

    char c = _tg.peek();                           ///< check EOF
    eof |= _tg.eof();                              /// * set EOF flag

    return cnt;
}

int Corpus::_get_images(int bid, int hdr, int bsz) {
    if (!data) DS_ALLOC(&data, bsz);               ///< allocate managed memory

    _ds.seekg(hdr + bid * bsz);                    /// * seek by batch id
    _ds.read((char*)data, bsz);                    /// * fetch batch images

    int cnt = _ds.gcount() / cell();               ///< # of sample fetched
    
    char c = _ds.peek();                           ///< check EOF
    eof |= _ds.eof();                              /// * set EOF flag

    return cnt;
}

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
