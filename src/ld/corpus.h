/**
 * @file
 * @brief Corpus class - NN corpus host-side interface object 
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __LD_CORPUS_H
#define __LD_CORPUS_H
#pragma once
#include <sstream>
#include <string>
#include "ten4_types.h"

#if  (T4_DO_OBJ && T4_DO_NN)

namespace t4::ld {
///
/// 1. Corpus data and label allocated from CUDA Managed Memory for debug
/// 2. moved to host heap if OK
/// 3. pre-fetching can be done in a separate thread
///
#define DS_ALLOC(p, sz)      H_ALLOC(p, sz)
#define IO_ERROR(fn)         ERROR("failed to open file %s\n", fn);

struct Corpus {
    const char *ds_name;     ///< data source name
    const char *tg_name;     ///< target label name
    
    U32 min;                 ///< range of the source data
    U32 max;
    U32 corpus_sz;           ///< number of total samples
    U32 batch_sz;            ///< number of samples of current mini-batch 
    
    U32 N, H, W, C;          ///< set dimensions and channel size
    union {
        U32 ctrl = 0;        ///< corpus control 
        struct {
            U32   eof  : 1;  ///< end of file control
            U32   xxx  : 31; ///< reserved
        };
    };
    U8 *data;                ///< source data pointer
    U8 *label;               ///< label data pointer
    
    Corpus(const char *data_name, const char *label_name, int min, int max)
        : ds_name(data_name), tg_name(label_name), min(min), max(max),
          corpus_sz(0), data(NULL), label(NULL) {}
    
    ~Corpus() {
        if (!data) return;

        cudaPointerAttributes attr;
        int host =
            cudaPointerGetAttributes(&attr, data)==cudaErrorInvalidValue &&
            attr.devicePointer==NULL;
        
        auto ds_free = [this, host](const char *name, U8 *p) {
            static const char *dev[] = { "CUDA Managed" , "HOST" };
            if (host) free(p);
            else      cudaFree(p);
            INFO("%s freed from %s memory", name, dev[host]);
        };
        ds_free(ds_name, data);
        if (label) ds_free(tg_name, label);
    }
    
    virtual U8 *operator [](int idx){ return &data[idx * cell()]; }  ///< data point
    
    int cell() { return H * W * C; }                                 ///< size of an element
    
    virtual Corpus *init(int mini_bsz, bool trace) { return NULL; }  ///< initialize dimensions
    virtual int    fetch(int bid, bool trace) { return 0; };         ///< load a mini-batch
    virtual Corpus *rewind()    { eof = 0; return this; }
    virtual Corpus *show(int n) { return this; }                     ///< show/preview n samples
};

} // namespace t4::ld

#endif // (T4_DO_OBJ && T4_DO_NN)
#endif // __LD_CORPUS_H

