/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Vu (Texture builder)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_VU_MNIST_VU_H
#define TEN4_SRC_VU_MNIST_VU_H
#include "vu.h"
///
/// MNIST GUI texture builder
///
class MnistVu : public Vu {
    int NX, NY;
public:
    __HOST__ MnistVu(Dataset &ds, int nx=40, int ny=30) :
        Vu(ds, ds.W * nx, ds.H * ny), NX(nx), NY(ny) {}
    
    __HOST__ virtual int init_host_tex();
};
#endif  // TEN4_SRC_VU_MNIST_VU_H



