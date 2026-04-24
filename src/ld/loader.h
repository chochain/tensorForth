/** -*- c++ -*-
 * @file
 * @brief Loader class - dataset loader factory interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __LDR_LOADER_H
#define __LDR_LOADER_H
#pragma once
#include "ten4_config.h"

#if (T4_DO_OBJ && T4_DO_NN)

#include "corpus.h"
#include "mu/dataset.h"

namespace t4::ld {

struct Loader {
    static void   init();
    static Corpus *get(mu::Dataset &ds, const char *ds_name=NULL);
};

} // namespace t4::ld

#endif  // (T4_DO_OBJ && T4_DO_NN)
#endif  // __LDR_LOADER_H


