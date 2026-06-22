/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __SYS_H
#define __SYS_H
#pragma once

#include "t4base.h"                             ///< include ten4_types.h
#include "util.h"                               ///< rand_opt
#include "io/aio.h"
#include "mu/mmu.h"

namespace t4     { class Debug;   }             ///< forward declaration
namespace t4::tb { class Summary; }

namespace t4 {
///
///@name System Manager Class
///@{
class System : public OnHost {                  ///< singleton class
    io::istr     &fin;                          ///< host input stream
    io::ostr     &fout;
    io::Istream  *_istr;                        ///< managed input stream
    io::Ostream  *_ostr;                        ///< managed output stream
    int          _khz;                          ///< GPU clock speed
    int          _trace;
    char         _tib[T4_STRBUF_SZ];            ///< terminal input buffer
    
    __HOST__ System(io::istr &i, io::ostr &o, int khz, int verbo);
    __HOST__ ~System();
    
public:
    mu::MMU      *mu;                           ///< memory management unit
    io::AIO      *io;                           ///< HOST IO manager
    tb::Summary  *tb = NULL;                    ///< TensorBoard SummaryWriter
    Debug        *db = NULL;                    ///< tracer (i.e. JTAG)
    ///
    /// singleton System controller
    ///
    static __HOST__ System *get_sys(io::istr &i, io::ostr &o, int khz, int verbo);
    static __HOST__ System *get_sys();          ///< singleton getter
    static __HOST__ void   free_sys();          ///< singleton destructor
    ///
    /// static System timing interfaces
    ///
    static __HOST__ DU   clock();
    static __HOST__ void delay(int ticks);
    ///
    /// Randomizer interfaces
    ///
    static __HOST__ void rand(DU d, rand_opt o);
    static __HOST__ void rand(DU *d, U64 sz, rand_opt o, DU bias=DU0, DU scale=DU1);
    ///
    /// TensorBoard support
    ///
    __HOST__ void        setup_tb(const char *tb_logdir, const char *tb_run_id);
    ///
    /// System functions
    ///
    __HOST__ int         readline(int hold);
    __HOST__ void        flush();                 ///< flush all events to output
    
    __HOST__ io::event  *_process_event(io::event *ev);
    __HOST__ io::event  *_process_opx(io::event *ev);
    __HOST__ io::event  *_process_tb(io::event *ev);
    ///
    /// debuging controls
    ///
    __HOST__ __INLINE__ int  &trace()       { return _trace; }
    __HOST__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    
    ///==============================================================================
    ///
    ///> System functions
    ///
    __HOST__  void op(OP op, DU n=DU0, U8 m=0, int i=0) {  ///< print operator
        *_ostr << io::opx(op, n, m, i);
    }
    __HOST__  void op_fn(char *fname) { *_ostr << fname; } ///< print filename
    
    __HOST__  void tbx(TB_OP op, char *tag, DU n=DU0, int i=0) { ///< tensorboard operator
        *_ostr << io::tbx(op, n, i);
        if (tag) *_ostr << tag;
    }
    ///
    /// input stream handler
    ///
    __HOST__  char key() {                                 ///< read key from console
        char c; *_istr >> c; return c;
    }
    __HOST__  char *scan(char delim)  {
        _istr->get_idiom(_tib, delim); return _tib;        ///< scan input stream for a given char
    }
    __HOST__  char *fetch() {                              ///< fetch next idiom
        return (*_istr >> _tib) ? _tib : NULL;
    }
    __HOST__  void clrbuf() { _istr->clear(); }
    ///
    /// output methods
    ///
    __HOST__  void spaces(int n) {                         ///< show spaces
        for (int i = 0; i < n; i++) _tib[i] = ' ';
        _tib[n] = '\0';
        *_ostr << _tib;
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

