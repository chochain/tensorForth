/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider host implemenation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "mnist.h"

#define LOG_COUNT 1000       /**< debug dump frequency */
#define MAX_BATCH 3          /**< debug, limit number of mini-batches */

Corpus *Mnist::init(int trace) {
    auto _u32 = [this](std::ifstream &fs) {
        U32 v = 0;
        char x;
        for (int i = 0; i < 4; i++) {
            fs.read(&x, 1);
            v <<= 8;
            v += (U32)*(U8*)&x;
        }
        return v;
    };
    if (_open()) return NULL;

    U32 X0, X1, N1=0;
    if (t_in) {
        X1 = _u32(t_in);    ///< label magic number 0x0801
        N1 = _u32(t_in);
        if (trace) INFO("\n\tMNIST label: magic=%08x => [%d]", X1, N1);
    }
    if (d_in) {
        X0 = _u32(d_in);    ///< image magic number 0x0803
        N  = _u32(d_in);
        H  = _u32(d_in);
        W  = _u32(d_in);
        C  = 1;
        if (trace)
            INFO("\n\tMNIST image: magic=%08x => [%d][%d,%d,%d]",
                 X0, N, H, W, C);
    }
    if (N != N1) {
        ERROR("Mnist::init label count %d != image count %d\n", N1, N);
        return NULL;
    }
    return this;
}

Corpus *Mnist::fetch(int bid, int n, int trace) {
    static int tick = 0;
    int bn = n * bid;                        ///< batch offset index
    if (eof || bn >= N) {                    /// * beyond total sample count?
        ERROR("Mnist::fetch EOF reached (needs rewind)\n");
        eof=1; return this;
    }
    ///
    /// fetch labels and images (and set eof if any of EOF reached)
    ///
    int b0   = _get_labels(bid, n);          ///< load batch labels
    batch_sz = _get_images(bid, n);          ///< load batch images
    if (b0 != batch_sz) {
        ERROR("Mnist::fetch #label=%d != #image=%d\n", b0, batch_sz);
        return NULL;
    }
    if (trace && (++tick == LOG_COUNT)) {
        INFO("\n\tMnist batch %d, loaded=%d/%d\n", bid, bn, N);
        if (trace > 1) _preview(n < 3 ? n : 3); /// * debug print
        tick = 0;
    }
    if (bn >= N) eof = 1;                       /// * EOF reached
    if (MAX_BATCH && ((bid+1) >= MAX_BATCH)) {  /// * forced stop? (debug)
        batch_sz = n >> 1;                      /// * fake a partial batch
        eof = 1;
    }
    return this;
}

int Mnist::_open() {
    if (ds_name) {
        d_in.open(ds_name, std::ios::binary);
        if (!d_in.is_open()) { IO_ERROR(ds_name); return -1; }
    }
    if (tg_name) {
        t_in.open(tg_name, std::ios::binary);
        if (!t_in.is_open()) { IO_ERROR(tg_name); return -1; }
    }
    return 0;
}

int Mnist::_close() {
    if (d_in.is_open()) d_in.close();
    if (t_in.is_open()) t_in.close();
    return 0;
}

int Mnist::_preview(int N) {
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

    return 0;
}

int Mnist::_get_labels(int bid, int n) {
    int hdr = sizeof(U32) * 2;                     ///< header to skip over
    int bsz = sizeof(U8) * n;                      ///< batch size

    if (!label) DS_ALLOC(&label, bsz);

    t_in.seekg(hdr + bid * bsz);                   /// * seek by batch
    t_in.read((char*)label, bsz);                  /// * fetch batch labels

    int cnt = t_in.gcount();                       ///< # of labels extracted

    char c = t_in.peek();                          ///< check EOF
    eof |= t_in.eof();                             /// * set EOF flag

    return cnt;
}

int Mnist::_get_images(int bid, int n) {
    int hdr = sizeof(U32) * 4;                     ///< header to skip over
    int bsz = n * cell();                          ///< image block size

    if (!data) DS_ALLOC(&data, bsz);

    d_in.seekg(hdr + bid * bsz);                   /// * seek by batch id
    d_in.read((char*)data, bsz);                   /// * fetch batch images

    int cnt = d_in.gcount() / cell();              ///< # of sample fetched
    
    char c = t_in.peek();                          ///< check EOF
    eof |= d_in.eof();                             /// * set EOF flag

    return cnt;
}

#endif // (T4_DO_OBJ && T4_DO_NN)
