
/**
 * @file
 * @brief tensorForth - Image Viewer class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_IMGVU_H
#define TEN4_IMGVU_H

#include "ten4_types.h"

typedef unsigned int TColor;

/////////////////////////////////////////////////////////////////////////////
// Filter configuration
/////////////////////////////////////////////////////////////////////////////
#define KNN_WINDOW_RADIUS 3
#define NLM_WINDOW_RADIUS 3
#define NLM_BLOCK_RADIUS  3
#define KNN_WINDOW_AREA \
  ((2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1))
#define NLM_WINDOW_AREA \
  ((2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1))
#define INV_KNN_WINDOW_AREA (1.0f / (float)KNN_WINDOW_AREA)
#define INV_NLM_WINDOW_AREA (1.0f / (float)NLM_WINDOW_AREA)

#define KNN_WEIGHT_THRESHOLD 0.02f
#define KNN_LERP_THRESHOLD   0.79f
#define NLM_WEIGHT_THRESHOLD 0.10f
#define NLM_LERP_THRESHOLD   0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif
#define iDivUp(a, b) ((((a) % (b)) != 0) ? ((a) / (b) + 1) : ((a) / (b)))


///=======================================================================
class ImgVu;

extern "C" int  gui_init(int *argc, char **argv, ImgVu *vu, int x, int y);
extern "C" int  gui_loop();

extern "C" void load_bmp(const char *fname, uchar4 **bmp_src, int *bmp_h, int *bmp_w);
///
/// in ${CUDA_HOME}/cuda-samples/Samples/2_Concepts_and_Techniques/imageDenoising/*.cuh
///
extern "C" void cuda_Copy(TColor *d_dst, int W, int H,
                          cudaTextureObject_t img);
extern "C" void cuda_KNN(TColor *d_dst, int W, int H, float Noise,
                         float lerpC, cudaTextureObject_t img);
extern "C" void cuda_KNNdiag(TColor *d_dst, int W, int H, float Noise,
                         float lerpC, cudaTextureObject_t img);
extern "C" void cuda_NLM(TColor *d_dst, int W, int H, float Noise,
                         float lerpC, cudaTextureObject_t img);
extern "C" void cuda_NLMdiag(TColor *d_dst, int W, int H, float Noise,
                         float lerpC, cudaTextureObject_t img);

extern "C" void cuda_NLM2(TColor *d_dst, int W, int H, float Noise,
                         float LerpC, cudaTextureObject_t img);
extern "C" void cuda_NLM2diag(TColor *d_dst, int W, int H, float Noise,
                         float LerpC, cudaTextureObject_t img);

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY     10  // ms
#define BUFFER_DATA(i)    ((char *)0 + i)

class ImgVu {
public:
    uchar4 *h_src;               ///< source image on host
    int    width, height;
    
    ImgVu(const char *fname);
    ~ImgVu() {
        free(h_src);
        cudaFreeArray(d_ary);
        GPU_CHK();
    }
    void   keyboard(unsigned char k);
    void   display(TColor *d_dst);

private:
    int   g_Kernel  = 0;
    bool  g_Diag    = false;
    
    float noiseStep = 0.025f;
    float lerpStep  = 0.025f;
    float knnNoise  = 0.32f;
    float nlmNoise  = 1.45f;
    float lerpC     = 0.2f;
    
    // CUDA array descriptor
    cudaArray           *d_ary;  ///< image on device
    cudaTextureObject_t img;     ///

    void   _img_flip(TColor *d_dst);
    void   _alloc_array();
};
#endif // TEN4_IMGVU_H
