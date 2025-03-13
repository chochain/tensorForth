/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __SYS_H
#define __SYS_H
#include <curand_kernel.h>
#include "debug.h"                              ///< include mmu/mmu.h, io/aio.h
///
///@name System Manager Class
///@{
class System : public Managed {                 ///< singleton class
    h_istr   &fin;                              ///< host input stream
    h_ostr   &fout;                             
    Istream  *_istr;                            ///< managed input stream
    Ostream  *_ostr;                            ///< managed output stream
    int      _trace;
    char     _pad[T4_STRBUF_SZ];                ///< terminal input buffer
    
    __HOST__ System(h_istr &i, h_ostr &o, int khz, int verbo);
    __HOST__ ~System();
    
public:
    MMU      *mu;                               ///< memory management unit
    AIO      *io;                               ///< HOST IO manager
    Debug    *db;                               ///< tracer (i.e. JTAG)
    ///
    /// singleton System controller
    ///
    static __HOST__ System *get_sys(h_istr &i, h_ostr &o, int khz, int verbo);
    static __HOST__ System *get_sys();          ///< singleton getter
    static __HOST__ void   free_sys();          ///< singleton destructor
    ///
    /// static System timing interfaces
    ///
    static __GPU__ DU   ms();
    static __GPU__ void delay(int ticks);
    static __GPU__ void rand(DU *d, U64 sz, rand_opt n, DU bias=DU0, DU scale=DU1);
    ///
    /// System functions
    ///
    __HOST__ int       readline(int hold);
    __HOST__ io_event  *process_event(io_event *ev);
    __HOST__ void      flush();
    ///
    /// debuging controls
    ///
    __BOTH__ __INLINE__ int  &trace()       { return _trace; }
    __BOTH__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    
    ///==============================================================================
    /// Device methods
    __GPU__  DU   rand(DU d, rand_opt n);                 ///< randomize a tensor
    ///
    ///> System functions
    ///
    __GPU__  void op(OP op, U8 m=0, DU n=DU0, int i=0) {  ///< print operator
        *_ostr << opx(op, m, n, i);
    }
    __GPU__  void op_fn(char *fname) { *_ostr << fname; } ///< print filename
    ///
    /// input stream handler
    ///
    __GPU__  char key() {                                 ///< read key from console
        char c; *_istr >> c; return c;
    }
    __GPU__  char *scan(char delim)  {
        _istr->get_idiom(_pad, delim); return _pad;       ///< scan input stream for a given char
    }
    __GPU__  char *fetch() {                              ///< fetch next idiom
        return (*_istr >> _pad) ? _pad : NULL;
    }
    __GPU__  void clrbuf() { _istr->clear(); }
    ///
    /// output methods
    ///
    __GPU__  void spaces(int n) {                         ///< show spaces
        for (int i = 0; i < n; i++) *_ostr << " ";
    }
    __GPU__  void dot(io_op o, DU v=DU0) {                ///< print literals
        switch (o) {
        case CR:    *_ostr << ENDL;                             break;
        case RDX:   *_ostr << setbase(INT(v));                  break;
        case DOT:   *_ostr << v << " ";                         break;
        case UDOT:  *_ostr << static_cast<U32>(v) << " ";       break;
        case EMIT:  { char b = (char)INT(v); *_ostr << b; }     break;
        case SPCS:  spaces(INT(v));                             break;
        default:    *_ostr << "unknown op=" << o; op(OP_FLUSH); break;
        }
    }
    __GPU__ void dotr(int w, DU v, int b, bool u=false) {
        *_ostr << setbase(b) << setw(w)
               << (u ? static_cast<U32>(v) : v);
    }
    __GPU__  void pstr(const char *str, io_op o=SPCS) {  ///< print string
        *_ostr << str;
        if (o==CR) *_ostr << ENDL;
    }
    __GPU__  void perr(const char *str, const char *msg) {
        *_ostr << str << msg << ENDL; op(OP_FLUSH);
    }
};
///@}
#endif // __SYS_H

