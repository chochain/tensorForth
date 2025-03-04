/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#if (!defined(__LDR_LOADER_H) && T4_DO_OBJ && T4_DO_NN)
#define __LDR_LOADER_H
#include "corpus.h"

struct Loader {
    static void   init(int trace);
    static Corpus *get(Dataset &ds, const char *ds_name=NULL);
};

#endif  // (!defined(__LDR_LOADER_H) && T4_DO_OBJ && T4_DO_NN)


