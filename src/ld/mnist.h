/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __LDR_MNIST_H
#define __LDR_MNIST_H
#pragma once
#include <iostream>
#include <fstream>                           /// std::ifstream
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

    virtual Corpus *init(int mini_bsz, bool trace);    ///< setup/check sizing
    virtual int    fetch(int bid, bool trace);         ///< fetch bid'th mini-batch
    virtual Corpus *rewind() {
        _ds.clear(); _tg.clear(); return Corpus::rewind();
    }
    virtual Corpus *show(int n);

private:
    std::ifstream _ds;                                 ///< data file handle
    std::ifstream _tg;                                 ///< target label file handle

    virtual int _open();                               ///< open data sources
    virtual int _close();                              ///< close data sources

    virtual int _get_labels(int bid);                  ///< load labels
    virtual int _get_images(int bid);                  ///< load images/data
};

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
#endif // __LDR_MNIST_H

