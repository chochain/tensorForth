/**
 * @file
 * @brief Vu class - CUDA Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_VU_H
#define T4_VU_H
#include <GL/gl.h>
#include "ten4_types.h"
#include "corpus.h"              /// in ../ldr

#define VUX(g)   GPU_ERR(g)      /**< check UI error */

typedef U32                         TColor;
typedef cudaGraphicsResource_t      cuGfxPbo;      /* cudaGraphicsResource pointer */
typedef cudaTextureObject_t         cuTexObj;      /* long long                    */

class Vu {
public:
    Corpus    &corpus;          ///< NN data source
    int       X, Y;             ///< view port dimensions
    uchar4    *h_tex   = NULL;  ///< host texture memory
    cudaArray *d_ary   = NULL;  ///< CUDA texture buffer on device
    cuGfxPbo  cu_pbo   = NULL;  ///< OpenGL-CUDA pixel buffer object handle
    cuTexObj  cu_tex   = 0;     ///< CUDA textrure object handle

    __HOST__ Vu(Corpus &cp, int x=0, int y=0);
    __HOST__ ~Vu();
    
    __HOST__ virtual void mouse(int button, int state, int x, int y) {}
    __HOST__ virtual void keyboard(U8 k)         {}
    __HOST__ virtual void display(TColor *d_dst) {}

private:
    __HOST__ void _init_host_tex();
    __HOST__ void _dump_host_tex();
    __HOST__ void _init_cuda_tex();
};

extern "C" int  gui_init(int *argc, char **argv);
extern "C" int  gui_add(Vu *vu);
extern "C" int  gui_loop();
    
#endif // T4_VU_H

