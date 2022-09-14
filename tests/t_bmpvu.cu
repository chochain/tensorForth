/** -*- c++ -*-
 * @File
 * @brief - tensorForth BMP Image Viewer implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "t_bmpvu.h"

BmpLoader &BmpLoader::load() {
    BMPHeader     hdr;
    BMPInfoHeader info;
    FILE          *fd;

    printf("Loading %s", d_fn);
    
    if (!(fd = fopen(d_fn, "rb"))) {
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
    C     = sizeof(uchar4);
    h_data= (U8*)malloc(W * H * C);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(info), SEEK_CUR);

    uchar4 *p = (uchar4*)h_data;
    int    z  = (4 - (3 * W) % 4) % 4;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++, p++) {
            p->z = fgetc(fd);
            p->y = fgetc(fd);
            p->x = fgetc(fd);
            p->w = 0xff;
        }
        for (int x = 0; x < z; x++) fgetc(fd);  // skip padding
    }
    if (ferror(fd)) {
        printf("***Unknown BMP load error.***\n");
        free(h_data);
        exit(EXIT_SUCCESS);
    }
    fclose(fd);
    printf(" => [%d,%d,%d] loaded\n", H, W, C);

    return *this;
}

__GPU__ __INLINE__ TColor make_color(float r, float g, float b, float a) {
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) << 8)  |
        ((int)(r * 255.0f) << 0);
}
__KERN__ void k_img_copy(TColor *dst, int W, int H, CuTexObj tex, bool flip) {
    const int j = threadIdx.x + blockDim.x * blockIdx.x;
    const int i = threadIdx.y + blockDim.y * blockIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = flip ? (float)(W - j) - 0.5f : (float)j + 0.5f;
    const float y = (float)i + 0.5f;

    if (j < W && i < H) {
        float4 v = tex2D<float4>(tex, x, y);
        dst[j + i * W] = make_color(v.x, v.y, v.z, 0);
    }
}

void BmpVu::_img_copy(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(X, Y, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, X, Y, cu_tex, false);
    GPU_CHK();
}
void BmpVu::_img_flip(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(X, Y, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, X, Y, cu_tex, true);
    GPU_CHK();
}

BmpVu::BmpVu(Dataset &ds) : Vu(ds) {
    uchar4 *p = h_tex;
    for (int i = 0; i < 10; i++) {
        printf("\n");
        for (int j = 0; j < 4; j++, p++) {
            printf("[%02x,%02x,%02x,%02x] ", p->x, p->y, p->z, p->w);
        }
    }
    printf("\n");
}

static const char *list_keys =
    "\nStarting GLUT main loop...\n"
    "Press [0] to view flipped image\n"
    "Press [1] to view original image\n"
    "Press [q] to exit\n";

int main(int argc, char **argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif
    printf("Starting %s...\n\n", argv[0]);

    cudaSetDevice(0);

    if (gui_init(&argc, argv)) return -1;

    BmpLoader &ldr = (*new BmpLoader("./data/portrait_noise.bmp")).load();
    BmpVu     &vu  = *new BmpVu(ldr);
    gui_add(vu);
    
    printf("%s", list_keys);
    
    return gui_loop();
}
