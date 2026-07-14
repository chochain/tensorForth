/**
 * @file
 * @brief Vu class - GL Image Viewer base class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VU_VU_H
#define __VU_VU_H
#pragma once
#include "ten4_types.h"
#include "render_source.h"       /// * IRenderSource - all gui.cu knows about Vu
#include "ld/corpus.h"           /// in ../ld

namespace t4::vu {

#define VUX(g)   GPU_ERR(g)      /**< check UI error */

typedef cudaTextureObject_t     cuTexObj;      /* long long */

class Vu : public IRenderSource {
    using Corpus = ld::Corpus;  ///< alias
public:
    Corpus    &corpus;          ///< NN data source
    int       X, Y;             ///< view port dimensions
    uchar4    *h_tex   = NULL;  ///< host texture memory
    cudaArray *d_ary   = NULL;  ///< CUDA texture buffer on device
    cuTexObj  cu_tex   = 0;     ///< CUDA textrure object handle

    __HOST__ Vu(Corpus &cp, int x=0, int y=0);
    __HOST__ ~Vu();

    __HOST__ int  width()  const override { return X; }
    __HOST__ int  height() const override { return Y; }
    __HOST__ virtual void mouse(int button, int state, int x, int y) override {}
    __HOST__ virtual void keyboard(U8 k)                              override {}
    __HOST__ virtual void display(TColor *d_dst)                      override {}

private:
    __HOST__ void _init_host_tex();
    __HOST__ void _dump_host_tex();
    __HOST__ void _init_cuda_tex();
};

} // namespace t4::vu

extern "C" int  gui_init(int *argc, char **argv);
extern "C" int  gui_add(t4::vu::IRenderSource *vu);
extern "C" int  gui_loop();

#endif // __VU_VU_H
