/** -*- c++ -*-
 * @File
 * @brief - tensorForth Image Loader with STB
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

typedef uint16_t U16;
typedef uint8_t  U8;

#define GAN_ERR(s) { \
    printf(" => ***GAN raw file load error: %s ***\n", (s)); \
    return -1;                                               \
}
int load_gan_img(const char *fname) {
    FILE *fd;
    
    printf("Loading %s", fname);

    if (!(fd = fopen(fname, "rb"))) GAN_ERR(fname);

    char   hdr[2];
    U16    shape[4];
    size_t sz = fread(hdr, sizeof(char), 2, fd);
    if (hdr[0]!='T' || hdr[1]!='4') {
        fclose(fd);
        GAN_ERR("file is not Ten4 raw format\n");
    }
        
    sz = fread(shape, 1, sizeof(shape), fd);
    int N = shape[3], H = shape[0], W = shape[1], C = shape[2];
    printf(" NHWC[%d,%d,%d,%d] loaded\n", N, H, W, C);
    ///
    /// STB use host memory, (compare to t_bmpvu which uses CUDA managed mem)
    ///
    /// +- N2*WC -+
    /// |         |
    /// |         | N2*H
    /// |         |
    /// +---------+
    ///
    int WC   = W * C;
    int HWC  = H * W * C;
    int N2   = static_cast<int>(ceilf(sqrtf(N)));
    int WX   = N2 * WC;
    int HX   = N2 * H;
    
    char *buf = (char*)malloc(HWC);
    char *img = (char*)malloc(HX * WX);

    for (int n = 0; n < N; n++) {
        sz = fread(buf, 1, HWC, fd);
        if (sz != HWC) {
            free(img);
            free(buf);
            GAN_ERR(" read size mismatched");
        }
        int x = n % N2, y = n / N2;
        for (int h = 0; h < H; h++) {
            char *p = img + (h + y * H) * WX + x * WC;
            memcpy(p, buf + h * WC, WC);
        }
        *(img + (y * H * WX) + x * WC) = (char)0xff;
    }
    fclose(fd);
    sprintf(buf, "%s_%dx%d.png", fname, N2, N2);
    printf(" => creating png[%d,%d] %s\n", WX, HX, buf);
    
    stbi_write_png(buf, N2 * W, N2 * H, C, img, WX);
    free(buf);
    free(img);

    return 0;
}
#include <iomanip>
#include <fstream>
using namespace std;

typedef unsigned short U16;

int gen_test_img(char *fname, U16 N, U16 H, U16 W, U16 C) {
    const char hdr[2] = { 'T', '4' };
    const int HWC = H * W * C;
    U16 shape[4] = { H, W, C, N };
    
    char *buf = (char*)malloc(HWC);
    
    ofstream fout(fname, ios_base::out | ios_base::binary);
    if (!fout.is_open()) {
        printf(" failed to open for output\n");
        return 1;
    }
    fout.write(hdr, sizeof(hdr));
    fout.write((const char*)shape, sizeof(shape));
    for (int n=0; n < N; n++) {
        for (int i=0; i < HWC; i++) {
            buf[i] = (char)(n * 2);
        }
        fout.write((const char*)buf, HWC);
    }
    free(buf);
    return 0;
}

int main(int argc, char **argv) {
    return argc > 2
        ? gen_test_img(argv[1], 100, 28, 28, 1)
        : load_gan_img(argv[1]);
}
