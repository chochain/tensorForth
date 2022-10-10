/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider host implemenation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mnist.h"

Corpus *Mnist::fetch(int batch_id, int batch_sz) {
    if (N == 0 && _setup()) return NULL;        /// * setup once only

    int bsz = batch_sz ? batch_sz : N;
    int b0  = _get_labels(batch_id, bsz);
    int b1  = _get_images(batch_id, bsz);
    if (b0 != b1) {
        fprintf(stderr, "ERROR: Mnist::load lable count != image count\n");
        return NULL;
    }
    _preview(bsz < 4 ? bsz : 4);               /// * debug print
    
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

int Mnist::_setup() {
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
    if (_open()) return -1;
    
    U32 X0, X1, N1=0;
    if (t_in) {
        X1 = _u32(t_in);    ///< label magic number 0x0801
        N1 = _u32(t_in);
        printf("\tMNIST label: magic=%08x => [%d]\n", X1, N1);
    }
    if (d_in) {
        X0 = _u32(d_in);    ///< image magic number 0x0803
        N  = _u32(d_in);
        H  = _u32(d_in);
        W  = _u32(d_in);
        C  = 1;
        printf("\tMNIST image: magic=%08x => [%d][%d,%d,%d]\n",
               X0, N, H, W, C);
    }
    if (N != N1) {
        fprintf(stderr, "ERROR: Mnist lable count != image count\n");
        return -2;
    }
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
                printf("%c%c", c, c1);                 // double width
            }
            printf("|");
        }
        printf("\n");
    }
    for (int n = 0; n < N; n++) {
        printf(" label=%-2d ", (int)label[n]);
        for (int j = 0; j < W*2 - 10; j++) printf("-");
        printf("+");
    }
    printf("\n");
    return 0;
}

int Mnist::_get_labels(int bid, int bsz) {
    int hdr = sizeof(U32) * 2;                     ///< header to skip over
    
    if (!label) DS_ALLOC(&label, bsz);

    t_in.seekg(hdr + bid * bsz);                   /// * seek by batch
    t_in.read((char*)label, bsz);                  /// * fetch batch labels

    int rst = t_in.eof() ? d_in.gcount() : bsz;
    printf("\tMnist.label batch[%d] sz=%d loaded\n", bid, rst);
    
    return rst;
}

int Mnist::_get_images(int bid, int bsz) {
    int hdr = sizeof(U32) * 4;                     ///< header to skip over
    int xsz = bsz * dsize();                       ///< image block size
    
    if (!data) DS_ALLOC(&data, xsz);

    d_in.seekg(hdr + bid * xsz);                   /// * seek by batch id
    d_in.read((char*)data, xsz);                   /// * fetch batch images

    int rst = d_in.eof() ? d_in.gcount() / dsize() : bsz;
    printf("\tMnist.image batch[%d] sz=%d loaded\n", bid, rst); 

    return rst;
}


