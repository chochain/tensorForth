/**
 * @file
 * @brief tensorForth - Dataset class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_DATASET_H
#define T4_DATASET_H

#define DS_ALLOC(p, sz) \
    if (cudaMallocManaged(p, sz) != cudaSuccess) { \
        fprintf(stderr, "ERROR: Dataset malloc %d\n", (int)(sz)); \
        exit(-1); \
    }
typedef uint8_t U8;

class Dataset {
public:
    const char *d_fn;         ///< data file name
    const char *t_fn;         ///< target lable file name
    
    int   N, H, W, C;         ///< dimensions and channel size
    int   batch_sz;
    
    U8    *data  = NULL;      ///< source data on host
    U8    *label = NULL;      ///< label data on host

    Dataset(const char *data_fn, const char *label_fn, int bsz=0)
        : d_fn(data_fn), t_fn(label_fn), batch_sz(bsz) {}
    ~Dataset() {
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
    int dsize() { return H * W * C; }
    int len()   { return N; }
    
    virtual Dataset *load() { printf("load() implemented?\n"); return this; }
    virtual U8 *operator [](int idx) { return &data[idx * dsize()]; }
};
#endif // T4_DATASET_H

