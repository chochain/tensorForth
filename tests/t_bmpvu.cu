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

#define BMP_ERR(s) { \
    printf(" => ***BMP load error: %s ***\n", (s)); \
    exit(EXIT_SUCCESS);                             \
}

BmpLoader *BmpLoader::load(int, int) {
    BMPHeader     hdr;
    BMPInfoHeader info;
    FILE          *fd;

    printf("Loading %s", ds_name);
    
    if (!(fd = fopen(ds_name, "rb"))) BMP_ERR("file access denied");
    
    size_t sz = fread(&hdr, sizeof(hdr), 1, fd);
    if (hdr.type != 0x4D42) BMP_ERR("bad file format");
    
    sz = fread(&info, sizeof(info), 1, fd);
    if (info.bitsPerPixel != 24) BMP_ERR("invalid color depth");
    if (info.compression) BMP_ERR("compressed image");

    W = info.width;
    H = info.height;
    C = sizeof(uchar4);
    ///
    /// CUDA managed memory, (compare to t_imgvu which using host mem)
    ///
    DS_ALLOC(&data, W * H * C);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(info), SEEK_CUR);

    uchar4 *p = (uchar4*)data;
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
        cudaFree(data);
        BMP_ERR("\n***Unknown BMP load error.***");
    }
    fclose(fd);
    printf(" => [%d,%d,%d] loaded\n", H, W, C);

    return this;
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
    dim3 grd(NGRID(X, Y, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, X, Y, cu_tex, false);
    GPU_CHK();
}
void BmpVu::_img_flip(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(NGRID(X, Y, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, X, Y, cu_tex, true);
    GPU_CHK();
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

    BmpLoader *ldr = (new BmpLoader("img/baboon.bmp"))->load();
    BmpVu     *vu  = new BmpVu(*ldr);
    gui_add(vu);
    
    printf("%s", list_keys);
    
    return gui_loop();
}
