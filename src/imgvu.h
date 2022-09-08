
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

class ImgVu;

extern "C" int  gui_init(int *argc, char **argv, ImgVu *vu, int x, int y);
extern "C" int  gui_loop();

extern "C" void load_bmp(const char *fname, uchar4 **bmp_src, int *bmp_h, int *bmp_w);

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

    void   _img_copy(TColor *d_dst);
    void   _img_flip(TColor *d_dst);
    void   _alloc_array();
};
#endif // TEN4_IMGVU_H
