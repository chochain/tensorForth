/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#pragma once
#include "ten4_config.h"

#if (!defined(__LDR_LOADER_H) && T4_DO_OBJ && T4_DO_NN)
#define __LDR_LOADER_H
#include "corpus.h"
#include "mmu/dataset.h"

namespace t4::ld {

struct Loader {
    static void   init();
    static Corpus *get(mu::Dataset &ds, const char *ds_name=NULL);
};

} // namespace t4::ld

#endif  // (!defined(__LDR_LOADER_H) && T4_DO_OBJ && T4_DO_NN)


