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
#include "ndata.h"              // in ../nn

typedef U32                         TColor;
typedef struct cudaGraphicsResource CuGfxPbo;
typedef cudaTextureObject_t         CuTexObj;

class Vu {
public:
    Ndata     &ndata;           ///< NN data source
    int       X, Y;             ///< view port dimensions
    uchar4    *h_tex   = NULL;  ///< texture on host
    
    cudaArray *d_ary   = NULL;  ///< CUDA texture buffer on device
    CuGfxPbo  *pbo     = NULL;  ///< OpenGL-CUDA exchange
    CuTexObj  cu_tex   = 0;     ///< cuda Textrure object
    
    __HOST__ Vu(Ndata &nd, int x=0, int y=0);
    __HOST__ virtual int  init_host_tex();   ///< rebuild texture buffer
    
    __HOST__ virtual void mouse(int button, int state, int x, int y) {}
    __HOST__ virtual void keyboard(U8 k)         {}
    __HOST__ virtual void display(TColor *d_dst) {}
    
    __HOST__ void tex_dump();
    __HOST__ void init_cuda_tex();
    __HOST__ void free_tex();
};

extern "C" int  gui_init(int *argc, char **argv);
extern "C" int  gui_add(Vu *vu);
extern "C" int  gui_loop();
    
#endif // T4_VU_H

