/**
 * @file
 * @brief Corpus class - NN corpus host-side interface object 
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_CORPUS_H
#define T4_CORPUS_H

#define DS_LOG1(...)         if (trace > 0) printf(__VA_ARGS__)
#define DS_ERROR(...)        fprintf(stderr, __VA_ARGS__)
#define IO_ERROR(fn)         fprintf(stderr, "ERROR: open file %s failed\n", fn)

#define DS_ALLOC(p, sz) \
    if (cudaMallocManaged(p, sz) != cudaSuccess) { \
        fprintf(stderr, "ERROR: Corpus malloc %d\n", (int)(sz)); \
        exit(-1); \
    }

typedef uint8_t U8;

struct Corpus {
    const char *ds_name;      ///< data source name
    const char *tg_name;      ///< target label name

    int   N, H, W, C;         ///< set dimensions and channel size
    int   eof    = 0;
    bool  trace  = false;
    U8    *data  = NULL;      ///< source data pointer
    U8    *label = NULL;      ///< label data pointer
    
    Corpus(const char *data_name, const char *label_name, bool trace)
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
    int dsize()   { return H * W * C; }                      ///< size of each point of data
    int len()     { return N; }                              ///< number of data point

    virtual Corpus *fetch(int batch_id=0, int batch_sz=0) {  /// * bid=bsz=0 => load entire set
        DS_LOG1("batch(U8*) implemented?\n");
        return this;
    }
    virtual Corpus *rewind() { eof = 0; return this; }
    virtual U8 *operator [](int idx){ return &data[idx * dsize()]; }  ///< data point
};
#endif // T4_CORPUS_H

