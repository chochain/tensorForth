/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __LDR_LOADER_H
#define __LDR_LOADER_H
#include "corpus.h"

struct Loader {
    static void   init(bool trace=0);
#if T4_ENABLE_OBJ
    static Corpus *get(Dataset &ds, const char *ds_name=NULL);
#endif  // T4_ENABLE_OBJ
};

#endif  // __LDR_LOADER_H


