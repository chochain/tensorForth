/** -*- c++ -*- 
 * @file
 * @brief Vu class - CUDA Image Viewer base class implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vu.h"

__HOST__
Vu::Vu(Corpus &cp, int x, int y) :
    corpus(cp), X(x ? x : corpus.W), Y(y ? y : corpus.H) {
    if (sizeof(uchar4) != 4) {
        fprintf(stderr, "ERR: Bad uchar4 size = %ld\n", sizeof(uchar4));
        exit(-1);
    }
    _init_host_tex();
    _dump_host_tex();
    _init_cuda_tex();
}

__HOST__
Vu::~Vu() {
    if (!d_ary) return;
    cudaDestroyTextureObject(cu_tex);     /// * release texture object
    cudaFreeArray(d_ary);                 /// * free device texture memory

    if (!h_tex) return;
    cudaFree(h_tex);                      /// * free host texture memory
}

__HOST__ void
Vu::_init_host_tex() {                    ///* shrink to fit
    size_t bsz = X * Y * sizeof(uchar4);  ///< buffer size
    VUX(cudaMallocManaged(&h_tex, bsz));  /// * h_tex on managed data
    if (!h_tex) {
        fprintf(stderr, "Vu.h_tex host texture malloc failed\n");
        exit(-1);
    }
    printf("Vu.h_tex=%p size=%ld", h_tex, bsz);
    
    int xs = (corpus.W + (X-1)) / X;      ///< x-stride
    int ys = (corpus.H + (Y-1)) / Y;      ///< y-stride
    int C  = corpus.C;                    ///< channel depth
    
    if (xs==1 && ys==1 && C==sizeof(uchar4)) {
        printf(", pass thru");
        VUX(cudaMemcpy(h_tex, corpus.data, bsz, cudaMemcpyHostToDevice));
        return;
    }
    printf(", resize");
    ///
    /// shrink to fit, TODO: interpolate (i.e. stb_image_resize)
    ///
    U8     *b = corpus.data;
    uchar4 *t = h_tex;
    for (int y = 0; y < Y; y++, b+=(ys - 1) * X * C) {
        for (int x = 0; x < X; x++, t++, b+=C) {
            t->x = *b;
            t->y = C < 2 ? *b   : *(b+1);
            t->z = C < 3 ? *b   : *(b+2);
            t->w = C < 4 ? 0xff : *(b+3);
        }
    }
}

__HOST__ void
Vu::_dump_host_tex() {
    if (!h_tex) return;
    uchar4 *p = h_tex;
    for (int y = 0; y < 10; y++) {
        printf("\n");
        for (int x = 0; x < 4; x++, p++) {
            printf("[%02x,%02x,%02x,%02x] ", p->x, p->y, p->z, p->w);
        }
    }
    printf("\n");
}

__HOST__ void
Vu::_init_cuda_tex() {
    int pitch = sizeof(uchar4) * X;
    
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
    VUX(cudaMallocArray(&d_ary, &fmt, X, Y));
    if (!d_ary) {
        fprintf(stderr, "Vu.d_ary managed array alloc failed\n");
        exit(-1);
    }
    printf(", Vu.d_ary=%p", d_ary);
    VUX(cudaMemcpy2DToArray(
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
  
    VUX(cudaCreateTextureObject(&cu_tex, &res, &desc, NULL));
    if (!cu_tex) {
        fprintf(stderr, "Vu.cu_tex cuda texture alloc failed\n");
        exit(-1);
    }
    printf(" ok\n");
}



