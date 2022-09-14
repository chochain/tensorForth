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
#include "dataset.h"

typedef U32 TColor;

class Vu {
public:
    Dataset             &dset;
    int                 X, Y;           ///< view port dimensions
    uchar4              *h_tex = NULL;  ///< texture on host
    cudaArray           *d_ary = NULL;  ///< texture on device
    cudaTextureObject_t img;            ///< cuda Textrure object
    struct cudaGraphicsResource *pbo;   ///< OpenGL-CUDA exchange
    
    Vu(Dataset &ds, int x, int y) : dset(ds), X(x), Y(y) {
        if (sizeof(uchar4) != 4) {
            printf("***Bad uchar4 size***\n");
            exit(EXIT_SUCCESS);
        }
        _build_texture();
        _bind_to_cuda();
    }
    ~Vu() {
        if (!h_tex) return;
        free(h_tex);
        
        if (!d_ary) return;
        cudaFreeArray(d_ary);           /// * free device memory
        GPU_CHK();
    }        
    virtual void mouse(int button, int state, int x, int y) {}
    virtual void keyboard(U8 k)         {}
    virtual void display(TColor *d_dst) {}
    
private:
    int  _build_texture();              ///< build texture memory
    void _bind_to_cuda();
};

extern "C" int  gui_init(int *argc, char **argv);
extern "C" int  gui_add(Vu &vu);
extern "C" int  gui_loop();
    
#endif // T4_VU_H

