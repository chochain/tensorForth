/** -*- c++ -*-
 * @file
 * @brief MnistVu class - MNIST NN Data Vu (Texture builder) interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_VU_MNIST_VU_H
#define TEN4_SRC_VU_MNIST_VU_H
#include "vu.h"

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
///
/// MNIST GUI texture builder
///
class MnistVu : public Vu {
    int NX, NY;
public:
    __HOST__ MnistVu(Corpus &cp, int nx=40, int ny=30) :
        Vu(cp, cp.W * nx, cp.H * ny), NX(nx), NY(ny) {
        _init_host_tex();
    }

private:
    __HOST__ virtual int _init_host_tex();
};

#endif  // (T4_ENABLE_OBJ && T4_ENABLE_NN)
#endif  // TEN4_SRC_VU_MNIST_VU_H



