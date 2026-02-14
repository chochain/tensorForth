/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Loader
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#define T4_DO_OBJ 1
#define T4_DO_NN  1

#include "loader.h"
#include "mnist_vu.h"

#define VU_W 40
#define VU_H 30
#define VU_N (VU_W*VU_H)

int main(int argc, char **argv) {
    const char *ds_name = "mnist_test";
    
    setenv("DISPLAY", ":0", 0);
    cudaSetDevice(0);

    Loader::init();
    if (gui_init(&argc, argv)) return -1;

    Dataset *ds = new Dataset(VU_N, 28, 28, 1);
    Corpus  *nd = Loader::get(*ds, ds_name);
    if (!nd) return 0;

    nd->init(true);
    printf("\n%s => [%d,%d,%d,%d]\n", ds_name, nd->N, nd->H, nd->W, nd->C);

    MnistVu *vu = new MnistVu(*nd->fetch(0, VU_N, 1), VU_W, VU_H);  // 1200 samples a shot
    gui_add(vu);
        
    return gui_loop();
}

