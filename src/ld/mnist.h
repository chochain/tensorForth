/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __LDR_MNIST_H
#define __LDR_MNIST_H
#pragma once
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "corpus.h"

namespace t4::ld {

///
/// MNIST NN data
///
class Mnist : public Corpus {
    
public:
    Mnist(const char *data_name, const char *label_name)
        : Corpus(data_name, label_name, 0, 256) {}
    ~Mnist() { _close(); }

    virtual Corpus *init(bool trace);                  ///< setup/check sizing
    virtual Corpus *fetch(int bid, int n, bool trace); ///< fetch bid'th mini-batch
};

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
#endif // __LDR_MNIST_H

