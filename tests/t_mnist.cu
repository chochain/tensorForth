/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Loader
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "loader.h"
#include "mnist_vu.h"

int main(int argc, char **argv) {
    setenv("DISPLAY", ":0", 0);
    cudaSetDevice(0);

    Loader::init(1);
    if (gui_init(&argc, argv)) return -1;
    
    Corpus *nd = Loader::get(0, "mnist_test");
    if (!nd) return 0;

    MnistVu *vu = new MnistVu(*nd->fetch(1200));  // 1200 samples a shot
    gui_add(vu);
        
    return gui_loop();
}

