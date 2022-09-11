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
#include "t_imgvu.h"

typedef unsigned char U8;

__GPU__ __INLINE__ TColor tex2color(float r, float g, float b, float a) {
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
        dst[j + i * W] = tex2color(v.x, v.y, v.z, 0);
    }
}

int ImgVu::_load() {
    printf("Loading %s...", fname);
    stbi_set_flip_vertically_on_load(true);
    U8 *img = stbi_load(fname, &W, &H, &N, 0);
    if (!img) {
        printf(" => failed\n");
        return -1;
    }
    uchar4 *x = h_src = (uchar4*)malloc(W * H * sizeof(uchar4));
    U8     *p = img;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++, x++) {
            x->x = *p++;
            x->y = *p++;
            x->z = *p++;
            x->w = 0x80;
        }
    }
    stbi_image_free(img);
    printf(" => [%d,%d,%d] loaded\n", W, H, N);
    
    Vu::setup();
    
    return 0;
}
void ImgVu::_img_copy(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(W, H, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, W, H, img, false);
    GPU_CHK();
}
void ImgVu::_img_flip(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(W, H, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, W, H, img, true);
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
    
    ImgVu *vu0 = new ImgVu("./data/cat_n_dog.jpg");
    ImgVu *vu1 = new ImgVu("./data/cat_n_dog.jpg");
    gui_add(vu0);
    gui_add(vu1);
    
    printf("%s", list_keys);
    
    return gui_loop();
    return 0;
}
