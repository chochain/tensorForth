/** -*- c++ -*-
 * @file
 * @brief Mnist class - MNIST dataset provider interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#if !defined(TEN4_SRC_LDR_MNIST_H) && (T4_ENABLE_OBJ && T4_ENABLE_NN)
#define TEN4_SRC_LDR_MNIST_H
#include <iostream>
#include <fstream>            // std::ifstream
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
    Mnist(const char *data_name, const char *label_name, bool trace)
        : Corpus(data_name, label_name, trace) {}
    ~Mnist() { _close(); }
    
    virtual Corpus *fetch(int batch_sz=0, int batch_id=0);  /// * bid=bsz=0 => load entire set
    virtual Corpus *rewind() { d_in.clear(); t_in.clear(); return Corpus::rewind(); }

private:
    int _open();
    int _close();
    int _setup();
    int _preview(int N);
    
    int _get_labels(int bsz, int bid);
    int _get_images(int bsz, int bid);
};
#endif // !defined(TEN4_SRC_LDR_MNIST_H) && (T4_ENABLE_OBJ && T4_ENABLE_NN)

