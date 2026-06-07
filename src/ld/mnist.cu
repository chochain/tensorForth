/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider host implemenation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <sstream>
#include <string>
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "mnist.h"

namespace t4::ld {
///
/// for init debug MAX_BATCH 3
///
//#define MAX_BATCH 3          /**< debug, limit number of mini-batches */
#define MAX_BATCH 0          /**< debug, limit number of mini-batches */

Corpus *Mnist::init(bool trace) {
    auto _u32 = [this](std::ifstream &fs) {
        U32 v = 0;
        char x;
        for (int i = 0; i < 4; i++) { /// Big-Endian
            fs.read(&x, 1);
            v <<= 8;
            v += (U32)*(U8*)&x;
        }
        return v;
    };
    if (_open()) return NULL;
    data  = NULL;
    label = NULL;
    min   = 0;
    max   = 256;

    U32 X0, X1, N1=0;
    if (_tg) {
        X1 = _u32(_tg);    ///< label magic number 0x0801
        N1 = _u32(_tg);
        if (trace) INFO("\tMNIST label: magic=%08x => [%d]\n", X1, N1);
    }
    if (_ds) {
        X0 = _u32(_ds);    ///< image magic number 0x0803
        N  = _u32(_ds);
        H  = _u32(_ds);
        W  = _u32(_ds);
        C  = 1;
        if (trace) INFO("\tMNIST image: magic=%08x => [%d][%d,%d,%d]\n",
              X0, N, H, W, C);
    }
    if (N != N1) {
        ERROR("Mnist::init label count %d != image count %d\n", N1, N);
        return NULL;
    }
    return this;
}

Corpus *Mnist::fetch(int bid, int n, bool trace) {
    int off = n * bid;                          ///< batch offset index
    if (eof || off >= N) {                      /// * beyond total sample count?
        ERROR("Mnist::fetch EOF reached (needs rewind)\n");
        eof=1; return this;
    }
    ///
    /// fetch labels and images (and set eof if any of EOF reached)
    ///
    int b0   = _get_labels(bid, n);             ///< load batch labels
    batch_sz = _get_images(bid, n);             ///< load batch images
    GPU_CHK();                                  /// * device sync after memory update
    
    if (b0 != batch_sz) {
        ERROR("Mnist::fetch #label=%d != #image=%d\n", b0, batch_sz);
        return NULL;
    }
    if ((off += batch_sz) >= N) eof = 1;        /// * EOF reached
    ///
    /// control partial batch for debugging
    ///
    if (MAX_BATCH && ((bid+1) >= MAX_BATCH)) {  /// * forced stop? (debug)
        batch_sz = n >> 1;                      /// * fake a partial batch
        eof = 1;
    }
    if (trace) {
        INFO("\tMnist batch[%d] loaded=%d/%d done=%d\n", bid, off, N, eof);
    }
    return this;
}

Corpus *Mnist::show(int N) {
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
int Mnist::_open() {
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

int Mnist::_close() {
    if (_ds.is_open()) _ds.close();
    if (_tg.is_open()) _tg.close();
    return 0;
}

int Mnist::_get_labels(int bid, int n) {
    int hdr = sizeof(U32) * 2;
    int bsz = n * sizeof(U8);
    
    if (!label) DS_ALLOC(&label, bsz);             ///< allocate managed memory

    _tg.seekg(hdr + bid * bsz);                    /// * seek by batch
    _tg.read((char*)label, bsz);                   /// * fetch batch labels

    int cnt = _tg.gcount();                        ///< # of labels extracted

    char c = _tg.peek();                           ///< check EOF
    eof |= _tg.eof();                              /// * set EOF flag

    return cnt;
}

int Mnist::_get_images(int bid, int n) {
    int hdr = sizeof(U32) * 4;
    int bsz = n * cell();
    
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
