#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imgvu.h"

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
ImgVu::ImgVu(const char *fname) {
    load_bmp(fname, &h_src, &H, &W);
    /*
    uchar4 *p = h_src;
    for (int i = 0; i < 10; i++) {
        printf("\n");
        for (int i = 0; i < 4; i++, p++) {
            printf("[%02x,%02x,%02x,%02x] ", p->x, p->y, p->z, p->w);
        }
    }
    */
    _alloc_array();
}

void ImgVu::_alloc_array() {
    cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&d_ary, &uchar4tex, W, H); GPU_CHK();
    cudaMemcpy2DToArray(
        d_ary, 0, 0,
        h_src, sizeof(uchar4) * W, sizeof(uchar4) * W, H,
        cudaMemcpyHostToDevice);
    GPU_CHK();

    cudaResourceDesc res;
    memset(&res, 0, sizeof(cudaResourceDesc));

    res.resType           = cudaResourceTypeArray;
    res.res.array.array   = d_ary;

    cudaTextureDesc desc;
    memset(&desc, 0, sizeof(cudaTextureDesc));

    desc.normalizedCoords = false;
    desc.filterMode       = cudaFilterModeLinear;
    desc.addressMode[0]   = cudaAddressModeWrap;
    desc.addressMode[1]   = cudaAddressModeWrap;
    desc.readMode         = cudaReadModeNormalizedFloat;
  
    cudaCreateTextureObject(&img, &res, &desc, NULL);
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

    const char *image_path = "./data/portrait_noise.bmp";
    ImgVu *vu = new ImgVu(image_path);

    if (gui_init(&argc, argv, vu, 512, 384)) return -1;
    
    printf("%s", list_keys);
    
    return gui_loop();
}
