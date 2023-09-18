/** -*- c++ -*-
 * @file
 * @brief MnistVu class - MNIST NN Data Vu (Texture builder) interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#if !defined(TEN4_SRC_VU_MNIST_VU_H) && (T4_ENABLE_OBJ && T4_ENABLE_NN)
#define TEN4_SRC_VU_MNIST_VU_H
#include "vu.h"
///
/// MNIST GUI texture builder
///
class MnistVu : public Vu {
    int NX, NY;
public:
    __HOST__ MnistVu(Corpus &cp, int nx=40, int ny=30) :
        Vu(cp, cp.W * nx, cp.H * ny), NX(nx), NY(ny) {}
    
    __HOST__ virtual int init_host_tex();
};
#endif  // !defined(TEN4_SRC_VU_MNIST_VU_H) && (T4_ENABLE_OBJ && T4_ENABLE_NN)



