/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#pragma once
#include "ten4_config.h"

#if (!defined(__LDR_MNIST_H) && T4_DO_OBJ && T4_DO_NN)
#define __LDR_MNIST_H
#include <iostream>
#include <fstream>                           /// std::ifstream
#include "corpus.h"

namespace t4::ld {

typedef uint8_t   U8;
typedef uint32_t  U32;
///
/// MNIST NN data
///
class Mnist : public Corpus {
    std::ifstream d_in;                      ///< data file handle
    std::ifstream t_in;                      ///< target label file handle
    
public:
    Mnist(const char *data_name, const char *label_name)
        : Corpus(data_name, label_name) {}
    ~Mnist() { _close(); }

    virtual Corpus *init();                  ///< setup/check sizing
    virtual Corpus *fetch(int bid, int n);   ///< fetch bid'th mini-batch
    virtual Corpus *show(int n);             ///< show/preview n samples
    virtual Corpus *rewind() { d_in.clear(); t_in.clear(); return Corpus::rewind(); }

private:
    int _open();
    int _close();
    
    int _get_labels(int bid, int n);
    int _get_images(int bid, int n);
};

} // namespace t4::ld
#endif // (!defined(__LDR_MNIST_H) && T4_DO_OBJ && T4_DO_NN)

