/** -*- c++ -*-
 * @File
 * @brief - tensorForth Dataset Loader (Factory)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <map>
#include "mnist.h"
#include "loader.h"

typedef std::map<const char*, Dataset*> DatasetMap;
DatasetMap ds_map;
///
/// TODO: to read from YAML config file
///
void Loader::init() {
    ds_map["mnist_train"] =
        new Mnist(
            "/u01/data/mnist/train-images-idx3-ubyte",
            "/u01/data/mnist/train-labels-idx1-ubyte");
    ds_map["mnist_test"] =
        new Mnist(
            "/u01/data/mnist/t10k-images-idx3-ubyte",
            "/u01/data/mnist/t10k-labels-idx1-ubyte");
}

Dataset *Loader::get(const char *ds_name, int batch_sz) {
    DatasetMap::iterator dsi = ds_map.find(ds_name);
    if (dsi == ds_map.end()) {
        printf("ERROR: Loader => %s not found\n", ds_name);
        return NULL;
    }
    return dsi->second->set_batch(batch_sz);
}


