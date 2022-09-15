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

typedef U32                         TColor;
typedef struct cudaGraphicsResource CuGfxPbo;
typedef cudaTextureObject_t         CuTexObj;

class Vu {
public:
    Dataset   &dset;
    int       X, Y;           ///< view port dimensions
    uchar4    *h_tex = NULL;  ///< texture on host
    
    cudaArray *d_ary = NULL;  ///< texture buffer on device
    CuGfxPbo  *pbo   = NULL;  ///< OpenGL-CUDA exchange
    CuTexObj  cu_tex = 0;     ///< cuda Textrure object
    
    __HOST__ Vu(Dataset &ds, int x=0, int y=0) :
        dset(ds), X(x ? x : ds.W), Y(y ? y : ds.H) {
        if (sizeof(uchar4) != 4) {
            printf("***Bad uchar4 size***\n");
            exit(EXIT_SUCCESS);
        }
        if (X == ds.W && Y == ds.H && ds.C == 4) {
            h_tex = (uchar4*)dset.h_data;   /// * pass thru, no buffer needed
        }
        else transform();                   /// * rebuild texture
        _bind_to_cuda();
    }
    __HOST__ ~Vu() {
        if (!h_tex) return;
        free(h_tex);

        cudaDestroyTextureObject(cu_tex);   /// * release texture object
        cudaFreeArray(d_ary);               /// * free device memory
        GPU_CHK();
    }
    __HOST__ virtual int  transform();      ///< rebuild texture buffer
    __HOST__ virtual void mouse(int button, int state, int x, int y) {}
    __HOST__ virtual void keyboard(U8 k)         {}
    __HOST__ virtual void display(TColor *d_dst) {}
    
private:
    __HOST__ void _bind_to_cuda();
};

extern "C" int  gui_init(int *argc, char **argv);
extern "C" int  gui_add(Vu &vu);
extern "C" int  gui_loop();
    
#endif // T4_VU_H

