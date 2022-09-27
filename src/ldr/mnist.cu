/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Provider (on host)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mnist.h"

Dataset *Mnist::load() {
    if (_open() || batch_id >= N) return NULL;
        
    int b0 = _load_labels();
    int b1 = _load_images();
    if (b0 != b1) {
        fprintf(stderr, "ERROR: Mnist::load lable count != image count\n");
        return NULL;
    }
    batch_id++;                   /// * bump current batch index
    
    for (int n = 0; n < 5; n++) {
        _preview((*this)[n], H, W, (int)label[n]);
    }
    return this;
}

Dataset *Mnist::get_batch(U8 *dst) {
    if (!this->load()) return NULL;
    
    int   bsz = batch_sz * dsize();
    float *d  = (float*)dst;
    U8    *s  = (U8*)data;
    for (int n = 0; n < bsz; n++) {
        *d++ = static_cast<float>(*s) / 256.0f;
    }
    return this;
}

U32 Mnist::_get_u32(std::ifstream &fs) {
    U32 v = 0;
    char x;
    for (int i = 0; i < 4; i++) {
        fs.read(&x, 1);
        v <<= 8;
        v += (U32)*(U8*)&x;
    }
    return v;
}

int Mnist::_open() {
    if (d_in.is_open() && t_in.is_open()) return 0;
    
    if (ds_name) {
        d_in.open(ds_name, std::ios::binary);
        if (!d_in.is_open()) { IO_ERROR(ds_name); return -1; }
    }
    if (tg_name) {
        t_in.open(tg_name, std::ios::binary);
        if (!t_in.is_open()) { IO_ERROR(tg_name); return -1; }
    }
    U32 X0, X1, N1=0;
    if (t_in) {
        X1 = _get_u32(t_in);    ///< label magic number 0x0801
        N1 = _get_u32(t_in);
        printf("MNIST label: magic=%08x => [%d]\n", X1, N1);
    }
    if (!batch_sz) batch_sz = N1;
    
    if (d_in) {
        X0 = _get_u32(d_in);    ///< image magic number 0x0803
        N  = _get_u32(d_in);
        H  = _get_u32(d_in);
        W  = _get_u32(d_in);
        C  = 1;
        printf("MNIST image: magic=%08x => [%d][%d,%d,%d]\n",
               X0, batch_sz, H, W, C);
    }
    if (N != N1) {
        fprintf(stderr, "ERROR: Mnist lable count != image count\n");
        return -2;
    }
    return 0;
}

int Mnist::_preview(U8 *img, int H, int W, int v) {
    static const char *map = " .:-=+*#%@";
    printf("\n+");
    for (int j = 0; j < W; j++) printf("--");
    for (int i = 0; i < H; i++) {
        printf("\n|");
        for (int j = 0; j < W; j++, img++) {
            char c = map[*img / 26];
            printf("%c%c", c, c);            // double width
        }
    }
    printf(" label=%d\n", v);
    return 0;
}

int Mnist::_load_labels() {
    int hdr = sizeof(U32) * 2;                     ///< header to skip over
    int bsz = batch_sz * sizeof(U8);               ///< block size
    
    if (!label) DS_ALLOC(&label, bsz);

    t_in.seekg(hdr + batch_id * bsz);              /// * seek by batch
    t_in.read((char*)label, N);                    /// * fetch batch labels

    int rst = t_in.eof() ? d_in.gcount() / (dsize() * sizeof(U8)): batch_sz;
    printf("Mnist.label batch[%d] sz=%d loaded\n", batch_id, rst);
    
    return rst;
}

int Mnist::_load_images() {
    int hdr = sizeof(U32) * 4;                     ///< header to skip over
    int bsz = batch_sz * H * W * C * sizeof(U8);   ///< block size
    
    if (!data) DS_ALLOC(&data, bsz);

    d_in.seekg(hdr + batch_id * bsz);              /// * seek by batch 
    d_in.read((char*)data, bsz);                   /// * fetch batch images

    int rst = d_in.eof() ? d_in.gcount() / (dsize() * sizeof(U8)): batch_sz;
    printf("Mnist.image batch[%d] sz=%d loaded\n", batch_id, rst); 

    return rst;
}


