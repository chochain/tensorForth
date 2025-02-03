/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_SYS_H
#define TEN4_SRC_SYS_H
#include <curand_kernel.h>
#include "debug.h"                              ///< include mmu/mmu.h, io/aio.h

#define ENDL '\n'

class System : public Managed {                 ///< singleton class
private:    
    int            _khz;                        ///< GPU clock speed
    curandState    *_seed;                      ///< for random number generator
    Istream        *_istr;                      ///< managed input stream
    Ostream        *_ostr;                      ///< managed output stream
    int            _trace;
    char           _pad[T4_STRBUF_SZ];          ///< terminal input buffer
    
public:
    MMU            *mu;                         ///< memory management unit
    AIO            *io;                         ///< HOST IO manager
    Debug          *db;
    
    __HOST__ System(h_istr &i, h_ostr &o, int khz, int verbo);
    __HOST__ ~System();
    ///
    /// System functions
    ///
    __HOST__ int       readline();
    __HOST__ io_event  *process_event(io_event *ev);
    __HOST__ void      flush();
    ///
    /// debuging controls
    ///
    __BOTH__ __INLINE__ int  khz()          { return _khz;   }
    __BOTH__ __INLINE__ int  trace()        { return _trace; }
    __BOTH__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    
    ///==============================================================================
    /// Device methods
    ///
    ///> System functions
    ///
    __GPU__  DU   ms() { return static_cast<double>(clock64()) / _khz; }
    __GPU__  DU   rand(DU d, rand_opt n) {                ///< randomize a tensor
#if T4_ENABLE_OBJ    
        if (IS_OBJ(d)) { random((Tensor&)du2obj(d), ntype); return d; }
#endif // T4_ENABLE_OBJ    
        return d * curand_uniform(&_seed[0]);
    }
    ///
    /// input stream handler
    ///
    __GPU__  char key() {                                 ///< read key from console
        char c; *_istr >> c; return c;
    }
    __GPU__  char *scan(char delim)  {
        _istr->get_idiom(_pad, delim); return _pad;       ///< scan input stream for a given char
    }
    __GPU__  char *fetch()           {                    ///< fetch next idiom
        return (*_istr >> _pad) ? _pad : NULL;
    }
//    __GPU__  void load(VM &vm, const char* fn);                   ///< load external Forth script
    ///
    /// output methods
    ///
    __GPU__  void spaces(int n) {                         ///< show spaces
        for (int i = 0; i < n; i++) *_ostr << " ";
    }
    __GPU__  void dot(io_op op, DU v=DU0) {               ///< print literals
        switch (op) {
        case RDX:   *_ostr << setbase(INT(v));                break;
        case CR:    *_ostr << ENDL;                           break;
        case DOT:   *_ostr << v << " ";                       break;
        case UDOT:  *_ostr << static_cast<U32>(v) << " ";     break;
        case EMIT:  { char b = (char)INT(v); *_ostr << b; }   break;
        case SPCS:  spaces(INT(v));                           break;
        default:    *_ostr << "unknown io_op=" << op << ENDL; break;
        }
    }
    __GPU__ void dotr(int w, DU v, int b, bool u=false) {
        *_ostr << setbase(b) << setw(w)
               << (u ? static_cast<U32>(v) : v);
    }
    __GPU__  void op(OP op, int a=0, DU n=DU0) {          ///< print operator
        *_ostr << opx(op, a, n);
    }
    __GPU__  void pstr(const char *str, io_op op=SPCS) {  ///< print string
        *_ostr << str;
        if (op==CR) { *_ostr << ENDL; }
    }
    __GPU__  void perr(const char *str, const char *msg) {
        *_ostr << str << msg << ENDL;
    }
};
#endif // TEN4_SRC_SYS_H

