/**
 * @file
 * @brief AIO class - asyn IO module implementation
 * @note
 *    AIO takes managed memory blocks as input and output buffers
 *    which can be access by both device and host
 *
 * <pre>Copyright (C) 2021 GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 */
#ifndef __IO_AIO_H
#define __IO_AIO_H
#include "istream.h"
#include "ostream.h"
#include "mmu/tensor.h"

typedef std::istream h_istr;          ///< host input stream
typedef std::ostream h_ostr;          ///< host output ostream

#define IO_DB(...)  { if (trace) INFO(__VA_ARGS__); }

class Model;
class Dataset;
class AIO {                           ///< create in host mode
    int &trace;
    
    __HOST__ AIO(int *verbo) : trace(*verbo) {}
    __HOST__ ~AIO() { TRACE("\\   AIO: instance freed\n"); }

public:
#if DO_MULTITASK
    static bool     io_busy;          ///< IO locking control
    static MUTEX    io;               ///< mutex for io access
    static COND_VAR cv_io;            ///< io control
    ///
    /// IO interface
    ///
    static void io_lock();            ///< lock IO
    static void io_unlock();          ///< unlock IO
#endif // DO_MULTITASK

    static __HOST__ AIO *get_io(int *verbo=NULL);          ///< assume AIO is instantiated
    static __HOST__ void free_io();

    __HOST__ void to_s(h_ostr &fs, DU v, int base);        ///< display value by ss_dump
    __HOST__ void print(h_ostr &fs, void *vp, U8 gt);
#if T4_DO_OBJ
    __HOST__ void to_s(h_ostr &fs, T4Base &t, bool view);  ///< display tensor by ss_dump
    __HOST__ void print(h_ostr &fs, T4Base &t);            ///< display in matrix format
    __HOST__ int  tsave(Tensor &t, char *fname, U8 mode);
    
#if T4_DO_NN    
    ///
    /// dataset IO methods
    ///
    __HOST__ int  dsfetch(Dataset &ds, char *ds_name=NULL, bool rewind=0); ///< fetch a dataset batch (rewind=false load batch)
    ///
    /// NN model persistence (i.e. serialization) methods
    ///
    __HOST__ int  nsave(Model &m, char *fname, U8 mode);
    __HOST__ int  nload(Model &m, char *fname, U8 mode, char *tib);
#endif // T4_DO_NN    
#endif // T4_DO_OBJ
    
private:
    int     _radix = 10;                       ///< output stream radix
    int     _thres = 10;                       ///< max cell count for each dimension
    int     _edge  = 3;                        ///< number of tensor edge items
    int     _prec  = 4;                        ///< shown floating point precision

#if T4_DO_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ///
    /// Tensor print methods
    ///
    __HOST__ void _print_vec(h_ostr &fs, DU *vd, U32 W, U32 C);
    __HOST__ void _print_mat(h_ostr &fs, DU *md, U32 *shape);
    __HOST__ void _print_tensor(h_ostr &fs, Tensor &t);
    ///
    /// Tensor persistence (i.e. serialization) methods
    ///
    __HOST__ int  _tsave_txt(h_ostr &fs, Tensor &t);
    __HOST__ int  _tsave_raw(h_ostr &fs, Tensor &t);
    __HOST__ int  _tsave_npy(h_ostr &fs, Tensor &t);
    
#if T4_DO_NN
    ///
    /// NN model print methods
    ///
    __HOST__ void _print_model(h_ostr &fs, Model &m);
    __HOST__ void _print_model_parm(h_ostr &fs, Tensor &in, Tensor &out);
    
    __HOST__ int  _nsave_model(h_ostr &fs, Model &m);
    __HOST__ int  _nsave_param(h_ostr &fs, Model &m);
    __HOST__ int  _nload_model(h_istr &fs, Model &m, char *fname, char *tib);
    __HOST__ int  _nload_param(h_istr &fs, Model &m);

#endif // T4_DO_NN
#endif // T4_DO_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
};

#endif // __IO_AIO_H
