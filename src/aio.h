/**
 * @file
 * @brief tensorForth Asyn IO module
 *
 * <pre>Copyright (C) 2021 GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 */
#ifndef TEN4_SRC_AIO_H_
#define TEN4_SRC_AIO_H_
#include "istream.h"
#include "ostream.h"
#include "mmu.h"

class AIO : public Managed {
public:
    Istream *_istr;         ///< managed input stream
    Ostream *_ostr;         ///< managed output stream
    MMU     *_mmu;          ///< memory managing unit
    int     _radix = 10;    ///< output stream radix
    bool    _trace;         ///< debug tracing control
    int     _edge  = 3;     ///< number of tensor edge items
    int     _precision = 4; ///< shown floating point precision

    AIO(MMU *mmu, bool trace) : _istr(new Istream()), _ostr(new Ostream()), _mmu(mmu), _trace(trace) {}

    __HOST__ Istream *istream() { return _istr; }
    __HOST__ Ostream *ostream() { return _ostr; }

    __HOST__ int  readline();
    __HOST__ void print_vec(DU *d, int mi, int ri);
    __HOST__ void print_mat(DU *d, int mi, int mj, int ri, int rj);
    __HOST__ void print_tensor(DU v);
    __HOST__ void print_node(obuf_node *node);
    __HOST__ void flush();
};
#endif // TEN4_SRC_AIO_H_
