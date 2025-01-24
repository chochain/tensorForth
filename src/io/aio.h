/**
 * @file
 * @brief AIO class - asyn IO module implementation
 *
 * <pre>Copyright (C) 2021 GreenII, this file is distributed under BSD 3-Clause License.</pre>
 *
 */
#ifndef TEN4_SRC_AIO_H
#define TEN4_SRC_AIO_H
#include "istream.h"
#include "ostream.h"
#include "mmu.h"            // in ../mmu

#define IO_TRACE(...)      { if (_mmu->trace()) INFO(__VA_ARGS__); }

typedef enum { RDX=0, CR, DOT, UDOT, EMIT, SPCS } io_op;

class AIO : public Managed {
public:
    Istream *_istr;                   ///< managed input stream
    Ostream *_ostr;                   ///< managed output stream
    MMU     *_mmu;                    ///< memory managing unit
    
    int     _radix = 10;              ///< output stream radix
    int     _thres = 10;              ///< max cell count for each dimension
    int     _edge  = 3;               ///< number of tensor edge items
    int     _prec  = 4;               ///< shown floating point precision

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
    
    AIO(MMU *mmu) :
        _istr(new Istream()), _ostr(new Ostream()), _mmu(mmu) {}

    __HOST__ Istream   *istream() { return _istr; }
    __HOST__ Ostream   *ostream() { return _ostr; }
    __HOST__ int       readline(std::istream &fin);
    __HOST__ obuf_node *process_node(std::ostream &fout, obuf_node *node);
    __HOST__ void      flush(std::ostream &fout);
    ///
    ///> IO functions
    ///
    __HOST__ void fin_setup(const char *line);
    __HOST__ void fout_setup(void (*hook)(int, const char*));

    __GPU__  char *scan(char c);                       ///< scan input stream for a given char
    __GPU__  int  fetch(string &idiom);                ///< read input stream into string
    __GPU__  char *word();                             ///< get next idiom
    
    __GPU__  char key();                               ///< read key from console
    __GPU__  void load(VM &vm, const char* fn);        ///< load external Forth script
    __GPU__  void spaces(int n);                       ///< show spaces
    __GPU__  void dot(io_op op, DU v=DU0);             ///< print literals
    __GPU__  void dotr(int w, DU v, int b, bool u=false); ///< print fixed width literals
    __GPU__  void pstr(const char *str, io_op op=SPCS);///< print string

private:
#if T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ///
    /// Tensor print methods
    ///
    __HOST__ void _print_obj(std::ostream &fout, DU v);
    __HOST__ void _print_vec(std::ostream &fout, DU *vd, int W, int C);
    __HOST__ void _print_mat(std::ostream &fout, DU *md, U16 *shape);
    __HOST__ void _print_tensor(std::ostream &fout, Tensor &t);
    ///
    /// Tensor persistence (i.e. serialization) methods
    ///
    __HOST__ int  _tsave(DU top, U16 mode, char *fname);
    __HOST__ int  _tsave_txt(std::ostream &fout, Tensor &t);
    __HOST__ int  _tsave_raw(std::ostream &fout, Tensor &t);
    __HOST__ int  _tsave_npy(std::ostream &fout, Tensor &t);
    
#if T4_ENABLE_NN
    ///
    /// NN model print methods
    ///
    __HOST__ void _print_model(std::ostream &fout, Model &m);
    __HOST__ void _print_model_parm(std::ostream &fout, Tensor &in, Tensor &out);
    ///
    /// dataset IO methods
    ///
    __HOST__ int  _dsfetch(DU id, U16 mode, char *ds_name=NULL); ///< fetch a dataset batch (more=true load batch, more=false rewind)
    ///
    /// NN model persistence (i.e. serialization) methods
    ///
    __HOST__ int  _nsave(DU top, U16 mode, char *fname);
    __HOST__ int  _nload(DU top, U16 mode, char *fname);
    
    __HOST__ int  _nsave_model(std::ostream &fout, Model &m);
    __HOST__ int  _nsave_param(std::ostream &fout, Model &m);
    __HOST__ int  _nload_model(std::istream &fin,  Model &m, char *fname);
    __HOST__ int  _nload_param(std::istream &fin,  Model &m);

#else
    __HOST__ void _print_model(std::ostream &fout, Model &m) {}  ///< stub
#endif // T4_ENABLE_NN
#else  // T4_ENABLE_OBJ
    __HOST__ void _print_obj(std::ostream &fout, DU v) {}        ///< stub

#endif // T4_ENABLE_OBJ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
};

#endif // TEN4_SRC_AIO_H
