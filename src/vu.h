/**
 * @file
 * @brief tensorForth - Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_VU_H
#define T4_VU_H
#include <GL/gl.h>
#include "ten4_types.h"

typedef U32 TColor;

class Vu {
public:
    const char          *fname;         ///< file name
    int                 N, W, H, C;     ///< dimensions and channel
    uchar4              *h_src = NULL;  ///< source image on host
    cudaArray           *d_ary = NULL;  ///< image on device
    cudaTextureObject_t img;            ///< cuda Textrure object
    struct cudaGraphicsResource *pbo;   ///< OpenGL-CUDA exchange
    
    Vu(const char *name);
    ~Vu();
    
    Vu &setup();                        ///< initialize cuda Texture
    
    virtual void mouse(int button, int state, int x, int y) {}
    virtual void keyboard(U8 k)         {}
    virtual void display(TColor *d_dst) {}
};

extern "C" int  gui_init(int *argc, char **argv);
extern "C" int  gui_add(Vu *vu);
extern "C" int  gui_loop();
    
#endif // T4_VU_H

