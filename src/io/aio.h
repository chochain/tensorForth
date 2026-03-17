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
#pragma once

#include "istream.h"
#include "ostream.h"
#include "mu/tensor.h"

namespace t4::nn { class Model;   }   /// forward declare
namespace t4::mu { class Dataset; }
namespace t4::io {

#define IO_DB(...)  { if (trace) INFO(__VA_ARGS__); }

class AIO {                           ///< create in host mode
    using Tensor  = mu::Tensor;       ///< aliases
    using Dataset = mu::Dataset;
    using Model   = nn::Model;
    
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

    static __HOST__ AIO *get_io(int *verbo=NULL);             ///< assume AIO is instantiated
    static __HOST__ void free_io();

    static __HOST__ void setfmt(h_ostr &o, void *vp);
    
    static __HOST__ std::string to_s(DU v, int base);         ///< display pure value
    static __HOST__ std::string to_s(void *vp, U8 gt);        ///< display value by type

#if T4_DO_OBJ
    static __HOST__ std::string to_s(T4Base &t, bool view);   ///< display tensor by ss_dump
    static __HOST__ std::string shape(T4Base &t);             ///< tensor shape
    
    __HOST__ std::string marshall(T4Base &t);                 ///< tensor to stream
    __HOST__ int  tsave(Tensor &t, char *fname, U8 mode);
    
#if T4_DO_NN    
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
    __HOST__ std::string _vec(DU *vd, U32 W, U32 C);
    __HOST__ std::string _mat(DU *md, U32 *shape);
    __HOST__ std::string _tensor(Tensor &t);
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
    __HOST__ std::string _model(Model &m);
    __HOST__ std::string _parm(Tensor &in, Tensor &out);
    ///
    /// Model persistence (i.e. serialization) methods
    ///
    __HOST__ int  _nsave_model(h_ostr &fs, Model &m);
    __HOST__ int  _nsave_param(h_ostr &fs, Model &m);
    __HOST__ int  _nload_model(h_istr &fs, Model &m, char *fname, char *tib);
    __HOST__ int  _nload_param(h_istr &fs, Model &m);

#endif // T4_DO_NN
#endif // T4_DO_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
};

} // namespace t4::io
#endif // __IO_AIO_H
