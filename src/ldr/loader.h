/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_LOADER_H
#define TEN4_LOADER_H
#include "corpus.h"

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)

struct Loader {
    static void   init(bool trace=0);
    static Corpus *get(int dset, const char *ds_name=NULL);
};

#endif  // (T4_ENABLE_OBJ && T4_ENABLE_NN)
#endif  // TEN4_LOADER_H


