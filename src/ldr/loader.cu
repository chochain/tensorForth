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
typedef std::map<int, Ndata*> DsetMap;
NdataMap nd_map;                             ///< string name, Ndata pair
DsetMap  ds_map;                             ///< Dataset, Ndata pair (cache)
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

Ndata *Loader::get(int dset, const char *ds_name) {
    DsetMap::iterator dsi = ds_map.find(dset);          /// * cache hit?
    if (dsi != ds_map.end()) return dsi->second;

    if (!ds_name) return NULL;                          /// * no name given

    NdataMap::iterator ndi = nd_map.find(ds_name);      /// * create new entry
    if (ndi == nd_map.end()) return NULL;

    return ds_map[dset] = ndi->second;
}


