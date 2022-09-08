/** -*- c++ -*-
 * @File
 * @brief - tensorForth BMP Image Viewer implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmpvu.h"

__GPU__ __INLINE__ TColor make_color(float r, float g, float b, float a) {
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) << 8)  |
        ((int)(r * 255.0f) << 0);
}
__KERN__ void k_img_copy(TColor *dst, int W, int H, cudaTextureObject_t img, bool flip) {
    const int j = threadIdx.x + blockDim.x * blockIdx.x;
    const int i = threadIdx.y + blockDim.y * blockIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = flip ? (float)(W - j) - 0.5f : (float)j + 0.5f;
    const float y = (float)i + 0.5f;

    if (j < W && i < H) {
        float4 v = tex2D<float4>(img, x, y);
        dst[j + i * W] = make_color(v.x, v.y, v.z, 0);
    }
}

void BmpVu::_load() {
    BMPHeader     hdr;
    BMPInfoHeader info;
    FILE *fd;

    printf("Loading %s...\n", fname);
    if (sizeof(uchar4) != 4) {
        printf("***Bad uchar4 size***\n");
        exit(EXIT_SUCCESS);
    }
    if (!(fd = fopen(fname, "rb"))) {
        printf("***BMP load error: file access denied***\n");
        exit(EXIT_SUCCESS);
    }
    
    fread(&hdr, sizeof(hdr), 1, fd);
    if (hdr.type != 0x4D42) {
        printf("***BMP load error: bad file format***\n");
        exit(EXIT_SUCCESS);
    }
    
    fread(&info, sizeof(info), 1, fd);
    if (info.bitsPerPixel != 24) {
        printf("***BMP load error: invalid color depth***\n");
        exit(EXIT_SUCCESS);
    }
    if (info.compression) {
        printf("***BMP load error: compressed image***\n");
        exit(EXIT_SUCCESS);
    }

    W     = info.width;
    H     = info.height;
    h_src = (uchar4*)malloc(info.width * info.height * 4);

    printf("BMP width: %u\n",  info.width);
    printf("BMP height: %u\n", info.height);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(info), SEEK_CUR);

    uchar4 *p = h_src;
    int z = (4 - (3 * info.width) % 4) % 4;
    for (int y = 0; y < info.height; y++) {
        for (int x = 0; x < info.width; x++, p++) {
            p->z = fgetc(fd);
            p->y = fgetc(fd);
            p->x = fgetc(fd);
        }
        for (int x = 0; x < z; x++) fgetc(fd);  // skip padding
    }
    if (ferror(fd)) {
        printf("***Unknown BMP load error.***\n");
        free(h_src);
        exit(EXIT_SUCCESS);
    }
    else printf("BMP file loaded successfully!\n");

    fclose(fd);
    printf("\nbmp %s[%d,%d] loaded", fname, info.height, info.width);
}

void BmpVu::_img_copy(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(W, H, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, W, H, img, false);
    GPU_CHK();
}
void BmpVu::_img_flip(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(W, H, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, W, H, img, true);
    GPU_CHK();
}
BmpVu::BmpVu(const char *fname) : Vu(fname) {
    _load();

    uchar4 *p = h_src;
    for (int i = 0; i < 10; i++) {
        printf("\n");
        for (int i = 0; i < 4; i++, p++) {
            printf("[%02x,%02x,%02x,%02x] ", p->x, p->y, p->z, p->w);
        }
    }
    Vu::setup();
}

static const char *list_keys =
    "\nStarting GLUT main loop...\n"
    "Press [0] to view flipped image\n"
    "Press [1] to view original image\n"
    "Press [q] to exit\n";

extern "C" int  gui_init(int *argc, char **argv, Vu *vu, int x, int y);
extern "C" int  gui_loop();

int main(int argc, char **argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif
    printf("Starting %s...\n\n", argv[0]);

    cudaSetDevice(0);

    const char *image_path = "./data/portrait_noise.bmp";
    BmpVu *vu = new BmpVu(image_path);

    if (gui_init(&argc, argv, vu, 512, 384)) return -1;
    
    printf("%s", list_keys);
    
    return gui_loop();
}
