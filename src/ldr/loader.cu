/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "loader.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include <map>
#include "mnist.h"
///
/// Note:
///   const char* key in map will not work because ptr1 != ptr2
///   but the string conversion slows it down by 3x.
///   We have only a few, so not too bad. Also we cache top <=> dataset
///
typedef std::map<std::string, Corpus*> CorpusMap;
typedef std::map<Dataset*, Corpus*> DsetMap;

CorpusMap _cp_map;                          ///< string name, Corpus pair
DsetMap   _ds_map;                          ///< Dataset, Corpus pair (cache)
Loader    _ldr;                             ///< loader
///
/// TODO: to read from YAML config file
///
void Loader::init() {
    _cp_map["mnist_train"] =
        new Mnist(
            "../data/MNIST/raw/train-images-idx3-ubyte",
            "../data/MNIST/raw/train-labels-idx1-ubyte");
    _cp_map["mnist_test"] =
        new Mnist(
            "../data/MNIST/raw/t10k-images-idx3-ubyte",
            "../data/MNIST/raw/t10k-labels-idx1-ubyte");
}

Corpus *Loader::get(Dataset &ds, const char *ds_name) {
    if (_cp_map.size()==0) init();                      /// * intialize if needed
    
    DsetMap::iterator dsi = _ds_map.find(&ds);          /// * cache hit?
    if (dsi != _ds_map.end()) return dsi->second;

    if (!ds_name) return NULL;                          /// * no name given

    CorpusMap::iterator cpi = _cp_map.find(ds_name);    /// * create new entry
    if (cpi == _cp_map.end()) return NULL;

    return _ds_map[&ds] = cpi->second;
}

#endif // (T4_DO_OBJ && T4_DO_NN)


