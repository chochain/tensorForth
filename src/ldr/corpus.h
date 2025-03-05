/**
 * @file
 * @brief Corpus class - NN corpus host-side interface object 
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_types.h"

#if (!defined(__LDR_CORPUS_H) && T4_DO_OBJ && T4_DO_NN)
#define __LDR_CORPUS_H

#define DS_LOG1(...)         { if (trace > 0) INFO(__VA_ARGS__); }
#define DS_ERROR(...)        fprintf(stderr, __VA_ARGS__)
#define IO_ERROR(fn)         fprintf(stderr, "ERROR: open file %s failed\n", fn)

#define DS_ALLOC(p, sz) \
    if (cudaMallocManaged(p, sz) != cudaSuccess) { \
        fprintf(stderr, "ERROR: Corpus malloc %d\n", (int)(sz)); \
        exit(-1); \
    }

struct Corpus {
    const char *ds_name;     ///< data source name
    const char *tg_name;     ///< target label name

    U32 N, H, W, C;          ///< set dimensions and channel size
    union {
        U32 ctrl = 0;        ///< corpus control 
        struct {
            U32   eof  : 1;  ///< end of file control
            U32   trace: 2;  ///< tracing level
            U32   xx   : 29; ///< reserved
        };
    };
    U8 *data  = NULL;        ///< source data pointer
    U8 *label = NULL;        ///< label data pointer
    
    Corpus(const char *data_name, const char *label_name, int trace)
       : ds_name(data_name), tg_name(label_name), trace(trace), N(0) {}
    
    ~Corpus() {
        if (!data) return;
        
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, data);
        if (attr.devicePointer != NULL) {
            DS_LOG1("free CUDA managed memory\n");
            cudaFree(data);
            if (label) cudaFree(label);
        }
        else {
            DS_LOG1("free HOST memory\n");
            free(data);
            if (label) free(label);
        }
    }
    int dsize()   { return H * W * C; }                    ///< size of each point of data
    int len()     { return N; }                            ///< number of data point
    
    virtual Corpus *init() { return NULL; }                /// * initialize dimensions
    virtual Corpus *fetch(int batch_id, int batch_sz=0) {  /// * bsz=0 => load entire set
        DS_LOG1("batch(U8*) implemented?\n");
        return this;
    }
    virtual Corpus *rewind() { eof = 0; return this; }
    virtual U8 *operator [](int idx){ return &data[idx * dsize()]; }  ///< data point
};

#endif // (!defined(__LDR_CORPUS_H) && T4_DO_OBJ && T4_DO_NN)

