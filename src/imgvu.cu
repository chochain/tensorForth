#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imgvu.h"

/////////////////////////////////////////////////////////////////////////////
// device/kernel functions
/////////////////////////////////////////////////////////////////////////////
__GPU__ float lerpf(float a, float b, float c) { return a + (b - a) * c; }
__GPU__ float vecLen(float4 a, float4 b) {
    return ((b.x - a.x) * (b.x - a.x) +
            (b.y - a.y) * (b.y - a.y) +
            (b.z - a.z) * (b.z - a.z));
}
__GPU__ TColor make_color(float r, float g, float b, float a) {
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
    dim3 grd(TGRID(width, height, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, width, height, img, false);
    GPU_CHK();
}
void ImgVu::_img_flip(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(width, height, 1, blk));

    k_img_copy<<<grd,blk>>>(d_dst, width, height, img, true);
    GPU_CHK();
}
ImgVu::ImgVu(const char *fname) {
    load_bmp(fname, &h_src, &height, &width);
    /*
    uchar4 *p = h_src;
    for (int i = 0; i < 10; i++) {
        printf("\n");
        for (int i = 0; i < 4; i++, p++) {
            printf("[%02x,%02x,%02x,%02x] ", p->x, p->y, p->z, p->w);
        }
    }
    */
    printf("\nbmp %s[%d,%d] loaded", fname, height, width);
    _alloc_array();
}

void ImgVu::keyboard(unsigned char k) {
    g_Kernel = (k == '0');
}
void ImgVu::display(TColor *d_dst) {
    if (g_Kernel) _img_flip(d_dst);
    else          _img_copy(d_dst);
}

void ImgVu::_alloc_array() {
    cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&d_ary, &uchar4tex, width, height); GPU_CHK();
    cudaMemcpy2DToArray(
        d_ary, 0, 0,
        h_src, sizeof(uchar4) * width, sizeof(uchar4) * width, height,
        cudaMemcpyHostToDevice);
    GPU_CHK();

    cudaResourceDesc res;
    memset(&res, 0, sizeof(cudaResourceDesc));

    res.resType = cudaResourceTypeArray;
    res.res.array.array = d_ary;

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

static const char *err_gui =
    "Error: failed to get minimal extensions for demo\n"
    "This sample requires:\n"
    "  OpenGL version 1.5\n"
    "  GL_ARB_vertex_buffer_object\n"
    "  GL_ARB_pixel_buffer_object\n";

int main(int argc, char **argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif
    printf("Starting %s...\n\n", argv[0]);

    cudaSetDevice(0);

    const char *image_path = "./data/portrait_noise.bmp";
    ImgVu *vu = new ImgVu(image_path);

    if (gui_init(&argc, argv, vu, 512, 384)) {
        fprintf(stderr, "%s", err_gui);
        return -1;
    }
    
    printf("%s", list_keys);
    
    return gui_loop();
}
