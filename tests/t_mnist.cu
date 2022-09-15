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

class MNIST : public Dataset {
public:
    MNIST(const char *data, const char *label) : Dataset(data, label) {}
    
    virtual MNIST &load() {
        int N0 = _load_labels();
        int N1 = _load_images();
        if (N0 != N1) {
            fprintf(stderr, "lable count != image count\n");
            exit(-1);
        }
        return *this;
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
int MNIST::_load_labels() {
    ifstream icin(t_fn, ios::binary);
    if (!icin.is_open()) return -2;
    
    U32 X = _u32(icin);
    N = _u32(icin);
    printf("MNIST label: magic=%08x => [%d]\n", X, N);
    h_label = new U8[N];
    for (int n = 0; n < N; n++) {
        icin.read((char*)h_label, N);
    }
    icin.close();
    return N;
}

int MNIST::_load_images() {
    ifstream icin(d_fn, ios::binary);
    if (!icin.is_open()) return -1;
    
    U32 X = _u32(icin);    ///< magic number
    U32 N = _u32(icin);
    H = _u32(icin);
    W = _u32(icin);
    C = 1;
    int dsz = H * W * C;

    printf("MNIST image: magic=%08x", X);
    h_data = (U8*)malloc(N * dsz);
    char *p = (char*)h_data;
    for (int n = 0; n < N; n++, p+=dsz) {
        icin.read(p, dsz);
    }
    icin.close();
    printf(" => [%d][%d,%d,%d], dsize=%d\n", N, H, W, C, dsz);
    
    for (int n = 0; n < 5; n++) {
        _preview((*this)[n], H, W, (int)h_label[n]);
    }
    return N;
}

class MnVu : public Vu {
    int NX, NY;
public:
    MnVu(Dataset &ds, int nx=40, int ny=30) :
        Vu(ds, ds.W * nx, ds.H * ny), NX(nx), NY(ny) {}
    
    __HOST__ virtual int init_host_tex() {
        auto fit = [this](int z0, U8 *src) {
            uchar4 *t = &h_tex[z0];
            for (int i = dset.H - 1; i >= 0; i--) {      // y top-down flip
                for (int j = 0; j < dset.W; j++, src++) {
                    int z = j + i * dset.W * NX;
                    t[z].x = t[z].y = t[z].z = *src;
                    t[z].w = 0xff;
                }
            }
        };
        for (int y = 0, y1 = NY-1; y < NY; y++, y1--) { // y1 top-down flip
            printf("\n");
            for (int x = 0; x < NX; x++) {
                int z   = x + y * NX;
                U8 *src = dset[z];
                int z0  = (x + y1 * dset.H * NX) * dset.W;
                fit(z0, src);
                printf("%1d", dset.h_label[z]);
            }
        }
        printf("\n");
        return 0;
    }
};

int main(int argc, char **argv) {
    setenv("DISPLAY", ":0", 0);
    cudaSetDevice(0);

    if (gui_init(&argc, argv)) return -1;

    MNIST  &ldr = (*new MNIST(
        "/u01/data/mnist/train-images-idx3-ubyte",
        "/u01/data/mnist/train-labels-idx1-ubyte")).load();
    MnVu &vu = *new MnVu(ldr);
    gui_add(vu);

    return gui_loop();
}

