/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Provider (on host)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mnist.h"

using namespace std;

Dataset *Mnist::load() {
    if (data) return this;
    
    int N0 = _load_labels();
    int N1 = _load_images();
    if (N0 != N1) {
        fprintf(stderr, "ERROR: Mnist::load lable count != image count\n");
        return NULL;
    }
    for (int n = 0; n < 5; n++) {
        _preview((*this)[n], H, W, (int)label[n]);
    }
    return this;
}

U32 Mnist::_get_u32(ifstream &fs) {
    U32 v = 0;
    char x;
    for (int i = 0; i < 4; i++) {
        fs.read(&x, 1);
        v <<= 8;
        v += (U32)*(U8*)&x;
    }
    return v;
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
    ifstream icin(t_fn, ios::binary);
    if (!icin.is_open()) return -2;
    
    U32 X = _get_u32(icin);
    N = _get_u32(icin);
    printf("MNIST label: magic=%08x => [%d]\n", X, N);
    
    DS_ALLOC(&label, N * sizeof(U8));

    for (int n = 0; n < N; n++) {
        icin.read((char*)label, N);
    }
    icin.close();
    return N;
}

int Mnist::_load_images() {
    ifstream icin(d_fn, ios::binary);
    if (!icin.is_open()) return -1;
    
    U32 X = _get_u32(icin);    ///< magic number
    U32 N = _get_u32(icin);
    H = _get_u32(icin);
    W = _get_u32(icin);
    C = 1;
    int dsz = H * W * C;

    printf("MNIST image: magic=%08x", X);

    DS_ALLOC(&data, dsz * N);
    char *p = (char*)data;
    for (int n = 0; n < N; n++, p+=dsz) {
        icin.read(p, dsz);
    }
    icin.close();
    printf(" => [%d][%d,%d,%d], dsize=%d\n", N, H, W, C, dsz);
    
    return N;
}


