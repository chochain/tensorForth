/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Provider (on host)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_LDR_MNIST_H
#define TEN4_SRC_LDR_MNIST_H
#include <iostream>
#include <fstream>         // std::ifstream
#include "../dataset.h"

typedef uint8_t   U8;
typedef uint32_t  U32;
///
/// MNIST dataset
///
class Mnist : public Dataset {
public:
    Mnist(const char *data, const char *label) : Dataset(data, label) {}
    
    virtual Dataset &load() {
        int N0 = _load_labels();
        int N1 = _load_images();
        if (N0 != N1) {
            fprintf(stderr, "lable count != image count\n");
            exit(-1);
        }
        return *this;
    }

private:
    U32 _get_u32(std::ifstream &fs);
    int _preview(U8 *img, int H, int W, int v);
    int _load_labels();
    int _load_images();
};
#endif // TEN4_SRC_LDR_MNIST_H

