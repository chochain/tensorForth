/**
 * @file
 * @brief tensorForth - Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_VU_H
#define T4_VU_H
#include "ten4_types.h"

typedef U32 TColor;

class Vu {
public:
    const char          *fname;         ///< file name
    int                 W, H;           ///< dimensions
    uchar4              *h_src = NULL;  ///< source image on host
    cudaArray           *d_ary = NULL;  ///< image on device
    cudaTextureObject_t img;            ///
    
    Vu(const char *name);
    ~Vu();
    
    Vu &setup();                        ///< initialize cuda Texture
    
    virtual void keyboard(U8 k) {}
    virtual void display(TColor *d_dst)    {}
};

#endif // T4_VU_H

