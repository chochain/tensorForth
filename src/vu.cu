/** -*- c++ -*- 
 * @File
 * @brief - tensorForth GUI - Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vu.h"

Vu::Vu(const char *name) : fname(name) {}
Vu::~Vu() {
    if (!h_src) return;
    
    free(h_src);
    cudaFreeArray(d_ary);
    GPU_CHK();
}

Vu &Vu::setup() {
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

    return *this;
}
