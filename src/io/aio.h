/**
 * @file
 * @brief AIO class - asyn IO module implementation
 *
 * <pre>Copyright (C) 2021 GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 */
#ifndef TEN4_SRC_AIO_H_
#define TEN4_SRC_AIO_H_
#include "istream.h"
#include "ostream.h"
#include "mmu.h"            // in ../mmu

class AIO : public Managed {
public:
    Istream *_istr;         ///< managed input stream
    Ostream *_ostr;         ///< managed output stream
    MMU     *_mmu;          ///< memory managing unit
    int     _radix = 10;    ///< output stream radix
    int     _thres = 10;    ///< max cell count for each dimension
    int     _edge  = 3;     ///< number of tensor edge items
    int     _prec  = 4;     ///< shown floating point precision
    int     _trace = 0;     ///< debug tracing control

    AIO(MMU *mmu, int verbose=0) :
        _istr(new Istream()), _ostr(new Ostream()), _mmu(mmu), _trace(verbose) {}

    __HOST__ Istream   *istream() { return _istr; }
    __HOST__ Ostream   *ostream() { return _ostr; }
    __HOST__ int       readline(std::istream &fin);
    __HOST__ obuf_node *process_node(std::ostream &fout, obuf_node *node);
    __HOST__ void      flush(std::ostream &fout);

private:
#if T4_ENABLE_OBJ
    ///
    /// object print methods
    ///
    __HOST__ void _print_obj(std::ostream &fout, DU v);
    __HOST__ void _print_vec(std::ostream &fout, DU *vd, int W, int C);
    __HOST__ void _print_mat(std::ostream &fout, DU *md, U16 *shape);
    __HOST__ void _print_tensor(std::ostream &fout, Tensor &t);
    __HOST__ void _print_model(std::ostream &fout, Model &m);
    ///
    /// dataset IO methods
    ///
    __HOST__ int  _dsfetch(DU top, U16 mode, char *ds_name=NULL); ///< fetch a dataset batch (more=true load batch, more=false rewind)
    ///
    /// Tensor & NN model persistence (i.e. serialization) methods
    ///
    __HOST__ int  _tsave(DU top, U16 mode, char *fname);
    __HOST__ int  _nsave(DU top, U16 mode, char *fname);
    __HOST__ int  _nload(DU top, U16 mode, char *fname);

    __HOST__ int  _tsave_txt(std::ostream &fout, Tensor &t);
    __HOST__ int  _tsave_raw(std::ostream &fout, Tensor &t);
    __HOST__ int  _tsave_npy(std::ostream &fout, Tensor &t);
    __HOST__ int  _nsave_model(std::ostream &fout, Model &m);
    __HOST__ int  _nsave_param(std::ostream &fout, Model &m);
    __HOST__ int  _nload_model(std::istream &fin,  Model &m, char *fname);
    __HOST__ int  _nload_param(std::istream &fin,  Model &m);
#endif // T4_ENABLE_OBJ
};
#endif // TEN4_SRC_AIO_H_
