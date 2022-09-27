/** -*- c++ -*- 
 * @File
 * @brief - tensorForth GUI - Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vu.h"

__HOST__
Vu::Vu(Ndata &nd, int x, int y) :
    ndata(nd), X(x ? x : ndata.W), Y(y ? y : ndata.H) {
    if (sizeof(uchar4) != 4) {
        fprintf(stderr, "ERR: Bad uchar4 size = %ld\n", sizeof(uchar4));
        exit(-1);
    }
    if (X == ndata.W && Y == ndata.H && ndata.C == 4) {
        h_tex = (uchar4*)ndata.data;       /// * pass thru, no buffer needed
        return;
    }
    ///
    /// malloc GL texture block (in uchar4 format)
    ///
    size_t bsz = X * Y * sizeof(uchar4);   ///< texture block size

    h_tex = (uchar4*)malloc(bsz);          ///< alloc texture block
    if (!h_tex) {
        fprintf(stderr, "Vu.h_tex malloc %ld bytes failed\n", bsz);
        exit(-1);
    }
}

__HOST__ int
Vu::init_host_tex() {
    int    C  = ndata.C;
    U8     *s = ndata.data;
    uchar4 *t = h_tex;
    for (int i = 0; i < Y; i++) {
        for (int j = 0; j < X; j++, t++, s+=C) {
            t->x = *s;
            t->y = C < 2 ? *s   : *(s+1);
            t->z = C < 3 ? *s   : *(s+2);
            t->w = C < 4 ? 0xff : *(s+3);
        }
    }
    return 0;                                ///< has malloc
}

__HOST__ void
Vu::tex_dump() {
#if CC_DEBUG
    if (!h_tex) return;
    uchar4 *p = h_tex;
    for (int i = 0; i < 10; i++) {
        printf("\n");
        for (int j = 0; j < 4; j++, p++) {
            printf("[%02x,%02x,%02x,%02x] ", p->x, p->y, p->z, p->w);
        }
    }
    printf("\n");
#endif // CC_DEBUG
}

__HOST__ void
Vu::init_cuda_tex() {
    int pitch = sizeof(uchar4) * X;
    
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
    CUX(cudaMallocArray(&d_ary, &fmt, X, Y));
    CUX(cudaMemcpy2DToArray(
        d_ary, 0, 0, h_tex, pitch, pitch, Y, cudaMemcpyHostToDevice));

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
  
    CUX(cudaCreateTextureObject(&cu_tex, &res, &desc, NULL));
}

__HOST__ void
Vu::free_tex() {
    if (!h_tex) return;
    
    CUX(cudaDestroyTextureObject(cu_tex));   /// * release texture object
    CUX(cudaFreeArray(d_ary));               /// * free device texture memory

    // free(h_tex);  /// * free host texture memory, TODO: => core dump?
}



