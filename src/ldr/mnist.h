/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_config.h"

#if (!defined(__LDR_MNIST_H) && T4_DO_OBJ && T4_DO_NN)
#define __LDR_MNIST_H
#include <iostream>
#include <fstream>            /// std::ifstream
#include "corpus.h"

using namespace std;

typedef uint8_t   U8;
typedef uint32_t  U32;
///
/// MNIST NN data
///
class Mnist : public Corpus {
    ifstream d_in;       ///< data file handle
    ifstream t_in;       ///< target label file handle
    
public:
    Mnist(const char *data_name, const char *label_name)
        : Corpus(data_name, label_name) {}
    ~Mnist() { _close(); }

    virtual Corpus *init(int trace);                    ///< setup/check sizing
    virtual Corpus *fetch(int bid, int n, int trace);   ///< fetch bid'th mini-batch
    virtual Corpus *rewind() { d_in.clear(); t_in.clear(); return Corpus::rewind(); }

private:
    int _open();
    int _close();
    int _preview(int N);
    
    int _get_labels(int bid, int n);
    int _get_images(int bid, int n);
};

#endif // (!defined(__LDR_MNIST_H) && T4_DO_OBJ && T4_DO_NN)

