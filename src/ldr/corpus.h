/**
 * @file
 * @brief Corpus class - NN corpus host-side interface object 
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_types.h"

#if (!defined(__LDR_CORPUS_H) && T4_DO_OBJ && T4_DO_NN)
#define __LDR_CORPUS_H

#define IO_ERROR(fn)         fprintf(stderr, "ERROR: open file %s failed\n", fn)
#define DS_ALLOC(p, sz)                                             \
    if (cudaMallocManaged(p, sz) != cudaSuccess) {                  \
        fprintf(stderr, "ERROR: Corpus malloc(%d) failed.\n", sz);  \
        exit(-1);                                                   \
    }

struct Corpus {
    const char *ds_name;     ///< data source name
    const char *tg_name;     ///< target label name

    U32 N, H, W, C;          ///< set dimensions and channel size
    union {
        U32 ctrl = 0;        ///< corpus control 
        struct {
            U32   eof  : 1;  ///< end of file control
            U32   xx   : 31; ///< reserved
        };
    };
    U8 *data  = NULL;        ///< source data pointer
    U8 *label = NULL;        ///< label data pointer
    
    Corpus(const char *data_name, const char *label_name)
        : ds_name(data_name), tg_name(label_name), N(0) {}
    
    ~Corpus() {
        if (!data) return;

        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, data);
        int host = attr.devicePointer==NULL;
        
        auto ds_free = [this, host](const char *name, U8 *p) {
            static const char *dev[] = { "CUDA Managed" , "HOST" };
            if (host) free(p);
            else      cudaFree(p);
            INFO("%s freed from %s memory", name, dev[host]);
        };
        ds_free(ds_name, data);
        if (label) ds_free(tg_name, label);
    }
    int dsize()   { return H * W * C; }                    ///< size of each point of data
    int len()     { return N; }                            ///< number of data point
    
    virtual Corpus *init(int trace) { return NULL; }                 /// * initialize dimensions
    virtual Corpus *fetch(int batch_id, int batch_sz, int trace) {   /// * bsz=0 => load entire set
        INFO("batch(U8*) implemented?\n");
        return this;
    }
    virtual Corpus *rewind() { eof = 0; return this; }
    virtual U8 *operator [](int idx){ return &data[idx * dsize()]; } ///< data point
};

#endif // (!defined(__LDR_CORPUS_H) && T4_DO_OBJ && T4_DO_NN)

