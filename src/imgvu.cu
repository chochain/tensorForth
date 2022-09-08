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
__KERN__ void k_img_flip(TColor *dst, int W, int H, cudaTextureObject_t img) {
    const int j = threadIdx.x + blockDim.x * blockIdx.x;
    const int i = threadIdx.y + blockDim.y * blockIdx.y;
    // Add half of a texel to always address exact texel centers
    const float x = (float)(W - j) - 0.5f;
    const float y = (float)i + 0.5f;

    if (j < W && i < H) {
        float4 v = tex2D<float4>(img, x, y);
        dst[j + i * W] = make_color(v.x, v.y, v.z, 0);
    }
}
void ImgVu::_img_flip(TColor *d_dst) {
    dim3 blk(T4_WARP_SZ, T4_WARP_SZ, 1);
    dim3 grd(TGRID(width, height, 1, blk));

    k_img_flip<<<grd,blk>>>(d_dst, width, height, img);
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
///
/// from ${CUDA_HOME}/cuda-samples/Samples/2_Concepts_and_Techniques/imageDenoising
///
#include <imageDenoising_copy_kernel.cuh>
#include <imageDenoising_knn_kernel.cuh>
#include <imageDenoising_nlm_kernel.cuh>
#include <imageDenoising_nlm2_kernel.cuh>

void ImgVu::keyboard(unsigned char k) {
    switch (k) {
    case '0': printf("Flip\n");  g_Kernel = 0; break;
    case '1': printf("Pass\n");  g_Kernel = 1; break;
    case '2': printf("KNN\n");   g_Kernel = 2; break;
    case '3': printf("NLM\n");   g_Kernel = 3; break;
    case '4': printf("NLM2\n");  g_Kernel = 4; break;
    case '*':
        printf(g_Diag ? "LERP highlighting mode.\n" : "Normal mode.\n");
        g_Diag = !g_Diag;
        break;
    case 'n': printf("Decrease noise level.\n");
        knnNoise -= noiseStep;
        nlmNoise -= noiseStep;
        break;
    case 'N':
        printf("Increase noise level.\n");
        knnNoise += noiseStep;
        nlmNoise += noiseStep;
        break;
    case 'l':
        printf("Decrease LERP quotient.\n");
        lerpC = MAX(lerpC - lerpStep, 0.0f);
        break;
    case 'L':
        printf("Increase LERP quotient.\n");
        lerpC = MIN(lerpC + lerpStep, 1.0f);
        break;
    case '?':
        printf(
            "lerpC=%.3f, knnNoise=%.3f, nlmNoise=%.3f\n",
            lerpC, knnNoise, nlmNoise);
        break;
    }
}
void ImgVu::display(TColor *d_dst) {
    float r = 1.0f;
    float c = lerpC;
    
    switch (g_Kernel) {
    case 0: _img_flip(d_dst);                     break;
    case 1: cuda_Copy(d_dst, width, height, img); break;
    case 2: r = 1.0f / (knnNoise * knnNoise);     break;
    case 3:
    case 4: r = 1.0f / (nlmNoise * nlmNoise);     break;
    }
    if (g_Diag) {
        switch (g_Kernel) {
        case 2: cuda_KNNdiag( d_dst, width, height, r, c, img); break;
        case 3: cuda_NLMdiag( d_dst, width, height, r, c, img); break;
        case 4: cuda_NLM2diag(d_dst, width, height, r, c, img); break;
        }
    }
    else {
        switch (g_Kernel) {
        case 2: cuda_KNN( d_dst, width, height, r, c, img); break;
        case 3: cuda_NLM( d_dst, width, height, r, c, img); break;
        case 4: cuda_NLM2(d_dst, width, height, r, c, img); break;
        }
    }
    GPU_CHK();
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
    "Press [1] to view noisy image\n"
    "Press [2] to view image restored with knn filter\n"
    "Press [3] to view image restored with nlm filter\n"
    "Press [4] to view image restored with modified nlm filter\n"
    "Press [*] to view smooth/edgy Ct's when a filter is active\n"
    "Press [?] to print Noise and Lerp Ct's\n"
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
