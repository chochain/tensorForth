/** -*- c++ -*-
 * @file
 * @brief Cifar10 class - CIFAR-10 dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __LDR_CIFAR10_H
#define __LDR_CIFAR10_H
#pragma once
#include <iostream>
#include <fstream>                           /// std::ifstream
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "corpus.h"

namespace t4::ld {
///
/// CIFAR-10 NN data
///
class Cifar10 : public Corpus {

#define LABEL_BSZ   1
#define IMAGE_H     32
#define IMAGE_W     32
#define IMAGE_C     3
#define IMAGE_HW    (IMAGE_H * IMAGE_W)
#define IMAGE_BSZ   (IMAGE_HW * IMAGE_C)
#define SAMPLE_BSZ  (LABEL_BSZ + IMAGE_BSZ)         /** 3073 */
    
public:
    Cifar10(const char *data_name) : Corpus(data_name, NULL, 0, 256) {}
    ~Cifar10() { _close(); }

    virtual Corpus *init(int mini_bsz, bool trace);    ///< setup/check sizing
    virtual int    fetch(int bid, bool trace);         ///< fetch bid'th mini-batch
    virtual Corpus *rewind() { _ds.clear(); return Corpus::rewind(); }

private:
    std::ifstream _ds;                                 ///< data file handle

    virtual int _open();                               ///< open data sources
    virtual int _close();                              ///< close data sources

    virtual int _get_data(int bid);                    ///< load labels
};

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
#endif // __LDR_CIFAR10_H

