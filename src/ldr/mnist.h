/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_LDR_MNIST_H
#define TEN4_SRC_LDR_MNIST_H
#include <iostream>
#include <fstream>            // std::ifstream
#include "ndata.h"

using namespace std;

typedef uint8_t   U8;
typedef uint32_t  U32;
///
/// MNIST NN data
///
class Mnist : public Ndata {
    ifstream d_in;       ///< data file handle
    ifstream t_in;       ///< target label file handle
    
public:
    Mnist(const char *data_name, const char *label_name)
        : Ndata(data_name, label_name) {}
    ~Mnist() { _close(); }
    
    virtual Ndata *load(int batch_sz=0, int batch_id=0);

private:
    int _open();
    int _close();
    int _setup();
    int _preview(U8 *img, int lbl);
    
    int _load_labels(int bsz, int bid);
    int _load_images(int bsz, int bid);
};
#endif // TEN4_SRC_LDR_MNIST_H

