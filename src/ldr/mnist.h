/** -*- c++ -*-
 * @File
 * @brief - tensorForth MNIST Dataset Provider (on host)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_LDR_MNIST_H
#define TEN4_SRC_LDR_MNIST_H
#include <iostream>
#include <fstream>            // std::ifstream
#include "../ndata.h"

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
    Mnist(const char *data_name, const char *label_name, int batch=0)
        : Ndata(data_name, label_name, batch_sz) {}
    ~Mnist() {
        if (d_in.is_open()) d_in.close();
        if (t_in.is_open()) t_in.close();
    }
    
    virtual Ndata *load();
    virtual Ndata *get_batch(U8 *dst);

private:
    U32 _get_u32(std::ifstream &fs);
    int _open();
    int _preview(U8 *img, int H, int W, int v);
    int _load_labels();
    int _load_images();
};
#endif // TEN4_SRC_LDR_MNIST_H

