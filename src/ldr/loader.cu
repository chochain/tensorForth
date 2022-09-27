/** -*- c++ -*-
 * @File
 * @brief - tensorForth Dataset Loader (Factory)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <map>
#include "mnist.h"
#include "loader.h"

typedef std::map<const char*, Ndata*> NdataMap;
NdataMap nd_map;
///
/// TODO: to read from YAML config file
///
void Loader::init() {
    nd_map["mnist_train"] =
        new Mnist(
            "/u01/data/mnist/train-images-idx3-ubyte",
            "/u01/data/mnist/train-labels-idx1-ubyte");
    nd_map["mnist_test"] =
        new Mnist(
            "/u01/data/mnist/t10k-images-idx3-ubyte",
            "/u01/data/mnist/t10k-labels-idx1-ubyte");
}

Ndata *Loader::get(const char *ds_name, int batch_sz) {
    NdataMap::iterator itr = nd_map.find(ds_name);
    return itr == nd_map.end()
        ? NULL
        : itr->second->set_batch(batch_sz);
}


