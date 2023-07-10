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
    int WC   = W * C;
    int HWC  = H * W * C;
    int WC10 = WC * 10;
    
    char *buf = (char*)malloc(HWC);
    char *img = (char*)malloc(N * HWC);

    for (int n = 0; n < N; n++) {
        sz = fread(buf, 1, HWC, fd);
        if (sz != HWC) {
            free(img);
            free(buf);
            GAN_ERR(" read size mismatched");
        }
        int x = n % 10, y = n / 10;
        for (int h = 0; h < H; h++) {
            char *p = img + ((h + y*H) * WC10 + x*WC);
            memcpy(p, buf, WC);
        }
    }
    fclose(fd);
    sprintf(buf, "%s_10x%d.png", fname, N/10);
    printf(" => creating 10x%d png %s\n", N/10, buf);
    
    stbi_write_png(buf, W*10, H*N/10, C, img, WC10);
    free(buf);
    free(img);

    return 0;
}

int main(int argc, char **argv) {
    return load_gan_img(argv[1]);
}
