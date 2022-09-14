/** -*- c++ -*- 
 * @File
 * @brief - tensorForth GUI - Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vu.h"

__HOST__ int
Vu::build_texture() {
    ///
    /// rebuild GL texture in uchar4 format
    /// 
    int c   = dset.C;
    int bsz = X * Y * sizeof(uchar4);        ///< texture block size

    h_tex = (uchar4*)malloc(bsz);            ///< alloc block
    if (!h_tex) return -1;
    
    U8     *s = dset.h_data;
    uchar4 *t = h_tex;
    for (int i = 0; i < Y; i++) {
        for (int j = 0; j < X; j++, t++, s+=c) {
            t->x = *s;
            t->y = c < 2 ? *s   : *(s+1);
            t->z = c < 3 ? *s   : *(s+2);
            t->w = c < 4 ? 0xff : *(s+3);
        }
    }
    _bind_to_cuda();
    return 1;                                ///< has malloc
}

__HOST__ void
Vu::_bind_to_cuda() {
    int pitch = sizeof(uchar4) * X;
    
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&d_ary, &fmt, X, Y); GPU_CHK();
    cudaMemcpy2DToArray(
        d_ary, 0, 0, h_tex, pitch, pitch, Y, cudaMemcpyHostToDevice);
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
  
    cudaCreateTextureObject(&cu_tex, &res, &desc, NULL);
    GPU_CHK();
}



