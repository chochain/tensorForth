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
#ifndef TEN4_SRC_AIO_H
#define TEN4_SRC_AIO_H
#include "istream.h"
#include "ostream.h"
#include "t4base.h"

#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
#include "dataset.h"                  // in ../mmu
#include "model.h"                    // in ../mmu
#include "loader.h"                   // in ../ldr (include corpus.h)
#endif // (T4_ENABLE_OBJ && T4_ENABLE_NN)

typedef std::istream h_istr;          ///< host input stream
typedef std::ostream h_ostr;          ///< host output ostream

#define IO_TRACE(...)      { if (_trace) INFO(__VA_ARGS__); }

class AIO {
public:
    friend class Debug;               ///< Debug can access my private members
    
    int     trace = 0;                ///< debug tracing verbosity level
    
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
    
    AIO(int verbo) : trace(verbo) {}

    __HOST__ int  to_s(DU s);
    __HOST__ int  to_s(IU w);
    __HOST__ void to_s(T4Base &t, bool view);

private:
    int     _radix = 10;              ///< output stream radix
    int     _thres = 10;              ///< max cell count for each dimension
    int     _edge  = 3;               ///< number of tensor edge items
    int     _prec  = 4;               ///< shown floating point precision

#if T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ///
    /// Tensor print methods
    ///
    __HOST__ void _print_obj(h_ostr &fout, DU v);
    __HOST__ void _print_vec(h_ostr &fout, DU *vd, int W, int C);
    __HOST__ void _print_mat(h_ostr &fout, DU *md, U16 *shape);
    __HOST__ void _print_tensor(h_ostr &fout, Tensor &t);
    ///
    /// Tensor persistence (i.e. serialization) methods
    ///
    __HOST__ int  _tsave(DU top, U16 mode, char *fname);
    __HOST__ int  _tsave_txt(h_ostr &fout, Tensor &t);
    __HOST__ int  _tsave_raw(h_ostr &fout, Tensor &t);
    __HOST__ int  _tsave_npy(h_ostr &fout, Tensor &t);
    
#if T4_ENABLE_NN
    ///
    /// NN model print methods
    ///
    __HOST__ void _print_model(h_ostr &fout, Model &m);
    __HOST__ void _print_model_parm(h_ostr &fout, Tensor &in, Tensor &out);
    ///
    /// dataset IO methods
    ///
    __HOST__ int  _dsfetch(Dataset &ds, U16 mode, char *ds_name=NULL); ///< fetch a dataset batch (more=true load batch, more=false rewind)
    ///
    /// NN model persistence (i.e. serialization) methods
    ///
    __HOST__ int  _nsave(Model &m, U16 mode, char *fname);
    __HOST__ int  _nload(Model &m, U16 mode, char *fname);
    
    __HOST__ int  _nsave_model(h_ostr &fout, Model &m);
    __HOST__ int  _nsave_param(h_ostr &fout, Model &m);
    __HOST__ int  _nload_model(h_istr &fin, Model &m, char *fname);
    __HOST__ int  _nload_param(h_istr &fin, Model &m);

#else
    __HOST__ void _print_model(h_ostr &fout, Model &m) {}  ///< stub
#endif // T4_ENABLE_NN
#else  // T4_ENABLE_OBJ
    __HOST__ void _print_obj(h_ostr &fout, DU v) {}        ///< stub

#endif // T4_ENABLE_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
};

#endif // TEN4_SRC_AIO_H
