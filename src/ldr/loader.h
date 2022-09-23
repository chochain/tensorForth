/** -*- c++ -*-
 * @File
 * @brief - tensorForth Dataset Loader (Factory)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_LOADER_H
#define TEN4_SRC_LOADER_H
#include "../dataset.h"

struct Loader {
    static void init();
    static Dataset *get(const char *ds_name);
};
#endif  // TEN4_SRC_LOADER_H


