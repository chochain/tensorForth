/** -*- c++ -*-
 * @File
 * @brief - tensorForth Dataset Loader (Factory)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_LOADER_H
#define TEN4_SRC_LOADER_H
#include "../ndata.h"

struct Loader {
    static void init();
    static Ndata *get(const char *ds_name, int batch_sz=0);
};
#endif  // TEN4_SRC_LOADER_H


