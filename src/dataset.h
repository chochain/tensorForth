/**
 * @file
 * @brief tensorForth - Dataset class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_DATASET_H
#define T4_DATASET_H

typedef unsigned char U8;

class Dataset {
public:
    const char *d_fn;              ///< data file name
    const char *t_fn;              ///< target lable file name
    int        N, H, W, C;         ///< dimensions and channel size
    U8         *h_data  = NULL;    ///< source data on host
    U8         *h_label = NULL;    ///< label data on host
    U8         *d_data  = NULL;    ///< source data on device

    Dataset(const char *data_fn, const char *label_fn) : d_fn(data_fn), t_fn(label_fn) {}
    ~Dataset() {
        if (!h_data) return;

        free(h_data);
        cudaFree(d_data);
        GPU_CHK();
    }
    int dsize() { return H * W * C; }
    int len()   { return N; }
    
    virtual Dataset &load() { printf("load() implemented?\n"); return *this; }
    virtual U8      *operator [](int idx) { return &h_data[idx * dsize()]; }
};
#endif // T4_DATASET_H

