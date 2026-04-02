/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __SYS_H
#define __SYS_H
#pragma once

#include <curand_kernel.h>
#include "debug.h"                              ///< include mmu/mmu.h, io/aio.h

namespace t4 {
///
///@name System Manager Class
///@{
class System : public Managed {                 ///< singleton class
    h_istr       &fin;                          ///< host input stream
    h_ostr       &fout;                             
    io::Istream  *_istr;                        ///< managed input stream
    io::Ostream  *_ostr;                        ///< managed output stream
    int          _trace;
    char         _pad[T4_STRBUF_SZ];            ///< terminal input buffer
    
    __HOST__ System(h_istr &i, h_ostr &o, int khz, int verbo);
    __HOST__ ~System();
    
public:
    mu::MMU  *mu;                               ///< memory management unit
    io::AIO  *io;                               ///< HOST IO manager
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
    static __BOTH__ DU   ms();
    static __HOST__ void delay(int ticks);
    static __GPU__  void rand(DU *d, U64 sz, rand_opt n, DU bias=DU0, DU scale=DU1);
    ///
    /// System functions
    ///
    __HOST__ int       readline(int hold);
    __HOST__ io_event  *process_event(io_event *ev);
    __HOST__ void      flush();
    ///
    /// debuging controls
    ///
    __HOST__ __INLINE__ int  &trace()       { return _trace; }
    __HOST__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    
    ///==============================================================================
    /// Device methods
    __HOST__  DU   rand(DU d, rand_opt n);                 ///< randomize a tensor
    ///
    ///> System functions
    ///
    __HOST__  void op(OP op, DU n=DU0, U8 m=0, int i=0) {  ///< print operator
        *_ostr << io::opx(op, n, m, i);
    }
    __HOST__  void op_fn(char *fname) { *_ostr << fname; } ///< print filename
    ///
    /// input stream handler
    ///
    __HOST__  char key() {                                 ///< read key from console
        char c; *_istr >> c; return c;
    }
    __HOST__  char *scan(char delim)  {
        _istr->get_idiom(_pad, delim); return _pad;       ///< scan input stream for a given char
    }
    __HOST__  char *fetch() {                              ///< fetch next idiom
        return (*_istr >> _pad) ? _pad : NULL;
    }
    __HOST__  void clrbuf() { _istr->clear(); }
    ///
    /// output methods
    ///
    __HOST__  void spaces(int n) {                         ///< show spaces
        for (int i = 0; i < n; i++) _pad[i] = ' ';
        _pad[n] = '\0';
        *_ostr << _pad;
    }
    __HOST__  void dot(io_op o, DU v=DU0) {                ///< print literals
        switch (o) {
        case CR:    *_ostr << ENDL;                             break;
        case RDX:   *_ostr << io::setbase(INT(v));              break;
        case DOT:   *_ostr << v << " ";                         break;
        case UDOT:  *_ostr << UINT(D2I(v)) << " ";              break;
        case EMIT:  { char b = (char)INT(v); *_ostr << b; }     break;
        case SPCS:  spaces(INT(v));                             break;
        default:    *_ostr << "unknown op=" << o; op(OP_FLUSH); break;
        }
    }
    __HOST__ void dotr(int w, DU v, int b, bool u=false) {
        *_ostr << io::setbase(b) << io::setw(w)
               << (u ? static_cast<U32>(v) : v);
    }
    __HOST__ void dots(int id, DU tos, int ss_idx, int base) {
        *_ostr << io::opx(OP_SS, tos, base, (id << 10) | ss_idx);
    }
    __HOST__  void pstr(const char *str, io_op o=SPCS) {  ///< print string
        *_ostr << str;
        if (o==CR) *_ostr << ENDL;
    }
    __HOST__  void perr(const char *str, const char *msg) {
        *_ostr << str << msg << ENDL; op(OP_FLUSH);
    }
};
///@}

} // namespace t4
#endif // __SYS_H

