/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
#include <map>
#include "loader.h"
#include "mnist.h"
///
/// Note:
///   const char* key in map will not work because ptr1 != ptr2
///   but the string conversion slows it down by 3x.
///   We have only a few, so not too bad. Also we cache top <=> dataset
///
typedef std::map<std::string, Corpus*> CorpusMap;
typedef std::map<Dataset&, Corpus*> DsetMap;

CorpusMap _cp_map;                          ///< string name, Corpus pair
DsetMap   _ds_map;                          ///< Dataset, Corpus pair (cache)
Loader    _ldr;                             ///< loader
///
/// TODO: to read from YAML config file
///
void Loader::init(int trace) {
    _cp_map["mnist_train"] =
        new Mnist(
            "../data/MNIST/raw/train-images-idx3-ubyte",
            "../data/MNIST/raw/train-labels-idx1-ubyte", trace);
    _cp_map["mnist_test"] =
        new Mnist(
            "../data/MNIST/raw/t10k-images-idx3-ubyte",
            "../data/MNIST/raw/t10k-labels-idx1-ubyte", trace);
}

Corpus *Loader::get(Dataset &ds, const char *ds_name) {
    if (_cp_map.size==0) init(T4_VERBOSE);              /// * intialize if needed
    
    DsetMap::iterator dsi = ds_map.find(ds);            /// * cache hit?
    if (dsi != ds_map.end()) return dsi->second;

    if (!ds_name) return NULL;                          /// * no name given

    CorpusMap::iterator cpi = cp_map.find(ds_name);     /// * create new entry
    if (cpi == cp_map.end()) return NULL;

    return ds_map[ds] = cpi->second;
}

#endif // (T4_ENABLE_OBJ && T4_ENABLE_NN)


