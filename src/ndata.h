/**
 * @file
 * @brief tensorForth - NN dataset class (host-side interface object)
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_NDATA_H
#define T4_NDATA_H

#define ND_ALLOC(p, sz) \
    if (cudaMallocManaged(p, sz) != cudaSuccess) { \
        fprintf(stderr, "ERROR: Ndata malloc %d\n", (int)(sz)); \
        exit(-1); \
    }
#define IO_ERROR(fn) \
    fprintf(stderr, "Ndata: fail to open file %s\n", fn);

typedef uint8_t U8;

struct Ndata {
    const char *ds_name;      ///< data source name
    const char *tg_name;      ///< target label name
    
    int   N, H, W, C;         ///< set dimensions and channel size
    int   batch_sz = 0;       ///< batch size (unit of data)
    int   batch_id = 0;       ///< current batch id
    
    U8    *data  = NULL;      ///< source data on host
    U8    *label = NULL;      ///< label data on host

    Ndata(const char *data_name, const char *label_name, int batch=0)
        : ds_name(data_name), tg_name(label_name), batch_sz(batch) {}
    
    ~Ndata() {
        if (!data) return;
        
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, data);
        if (attr.devicePointer != NULL) {
            printf("free CUDA managed memory\n");
            cudaFree(data);
            if (label) cudaFree(label);
        }
        else {
            printf("free HOST memory\n");
            free(data);
            if (label) free(label);
        }
    }
    Ndata *set_batch(int bsz, int idx=0) {
        batch_sz = bsz;
        batch_id = idx;
        return this;
    }
    int dsize() { return H * W * C; }
    int len()   { return N; }
    
    virtual Ndata *load() {
        printf("load() implemented?\n");
        return this;
    }
    virtual Ndata *get_batch(U8 *dst) {
        printf("batch(U8*) implemented?\n");
        return this;
    }
    virtual U8 *operator [](int idx){ return &data[idx * dsize()]; }
};
#endif // T4_NDATA_H

