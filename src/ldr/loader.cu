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

void Loader::init() {
    ds_map["mnist_train"] =
        new Mnist(
            "/u01/data/mnist/train-images-idx3-ubyte",
            "/u01/data/mnist/train-labels-idx1-ubyte");
    ds_map["mnist_test"] =
        new Mnist(
            "/u01/data/mnist/test-images-idx3-ubyte",
            "/u01/data/mnist/test-labels-idx1-ubyte");
}

Dataset *Loader::get(const char *ds_name) {
    DatasetMap::iterator it = ds_map.find(ds_name);
    if (it == ds_map.end()) {
        printf("ERROR: Loader => %s not found\n", ds_name);
        return NULL;
    }
    else {
        Dataset *ds = it->second;
        return ds->load();
    }
}


