/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Loader
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iostream>    // std::cout
#include <fstream>     // std::ifstream
#include "../src/vu.h"

using namespace std;

typedef unsigned int  U32;
typedef unsigned char U8;

class MNIST : public Vu {
public:
    U8 *labels = NULL;
    
    MNIST(const char *path) : Vu(path) {
        _load_labels();
        _load_images();
        Vu::setup();
    }
    
private:
    U32 _u32(ifstream &fs);
    int _preview(U8 *img, int H, int W, int v);
    
    int _load_labels();
    int _load_images();
};

U32 MNIST::_u32(ifstream &fs) {
        U32 v = 0;
        char x;
        for (int i = 0; i < 4; i++) {
            fs.read(&x, 1);
            v <<= 8;
            v += (U32)*(U8*)&x;
        }
        return v;
}
int MNIST::_preview(U8 *img, int H, int W, int v) {
    static const char *map = " .:-=+*#%@";
    for (int i = 0; i < H; i++) {
        printf("\n");
        for (int j = 0; j < W; j++) {
            printf("%c", *(map + (*img++ / 26)));
        }
    }
    printf(" label=%d\n", v);
    return 0;
}
int MNIST::_load_images() {
    ifstream icin("/u01/data/mnist/train-images-idx3-ubyte", ios::binary);
    if (!icin.is_open()) return -1;
    
    U32 _X = _u32(icin);
    U32 _N = _u32(icin);
    U32 _H = _u32(icin);
    U32 _W = _u32(icin);

    printf("MNIST image: magic=%08x,[%d][%d,%d]\n", _X, _N, _H, _W);
    U8 **img_lst = new U8*[_N];
    for (int n = 0; n < _N; n++) {
        img_lst[n] = new U8[_H * _W];
        icin.read((char*)img_lst[n], _H * _W);
    }
    icin.close();
/*
    for (int n = 0; n < 5; n++) {
        _preview(img_lst[n], H, W, (int)labels[n]);
    }
*/
    auto fill = [this](U8 *p, int z0, int h, int w) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++, p++) {
                int z = z0 + j + ((h-i-1) * w * 40);   // flip top-down
                h_src[z].x = h_src[z].y = h_src[z].z = *p;
                h_src[z].w = 0xff;
            }
        }
    };
    h_src = (uchar4*)malloc(1200 * _H * _W * sizeof(uchar4));
    for (int ny = 0; ny < 30; ny++) {
        printf("\n");
        for (int nx = 0; nx < 40; nx++) {
            U8 *p  = img_lst[nx + ny * 40];
            int z0 = (nx + (30-ny-1) * _H * 40) * _W;  // flip top-down
            fill(p, z0, _H, _W);
            printf("%1d", labels[nx + ny * 40]);
        }
    }
    printf("\n");
    C = 4;
    H = _H * 30;
    W = _W * 40;
    for (int n = 0; n < _N; n++) delete[] img_lst[n];
    delete[] img_lst;
    printf(" => [%d,%d,%d]\n", H, W, C);
    
    return 0;
}

int MNIST::_load_labels() {
    ifstream icin("/u01/data/mnist/train-labels-idx1-ubyte", ios::binary);
    if (!icin.is_open()) return -2;
    
    U32 X = _u32(icin);
    U32 N = _u32(icin);
    printf("MNIST label: magic=%08x,[%d]\n", X, N);
    labels = new U8[N];
    for (int n = 0; n < N; n++) {
        icin.read((char*)labels, N);
    }
    icin.close();
    return 0;
}

int main(int argc, char **argv) {
    setenv("DISPLAY", ":0", 0);
    cudaSetDevice(0);

    if (gui_init(&argc, argv)) return -1;

    MNIST *vu = new MNIST("/u01/data/mnist");
    gui_add(vu);

    return gui_loop();
    return 0;
}

