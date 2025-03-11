/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __DEBUG_H
#define __DEBUG_H
#include <iomanip>                                        /// setw, setprec, setbase...
#include "t4base.h"                                       ///< include ten4_types.h
#include "io/aio.h"
#include "vm/param.h"
#include "mmu/mmu.h"

#define ENDL    '\n'
///
///@name Debugger/Tracer class
///@{
class Debug {                                             ///< friend class to MMU and AIO
    MMU      *mu;                                         ///< memory controller
    AIO      *io;                                         ///< streaming io controller
    h_ostr   &fout;                                       ///< host output stream
    
    char     tmp[256];                                    ///< tmp string buffer
    
    __HOST__ Debug(h_ostr &o) : fout(o) {
        mu = MMU::get_mmu();
        io = AIO::get_io();
    }
    __HOST__ ~Debug() { TRACE("\\   Debug: instance freed\n"); }
    
public:
    static __HOST__ Debug *get_db(h_ostr &o);             ///< singleton contructor
    static __HOST__ Debug *get_db();                      ///< singleton getter
    static __HOST__ void  free_db();                      ///< singleton destructor
    
    __HOST__ void keep_fmt();
    __HOST__ void reset_fmt();

    __HOST__ void print(void *vp, U8 gt);                      ///< display value, io proxy
    __HOST__ void ss_dump(IU id, int sz, DU tos, int base=10); ///< show data stack content
    __HOST__ void words();                                     ///< list dictionary words
    __HOST__ void mem_dump(IU addr, int sz);                   ///< dump memory frm addr...addr+sz
    __HOST__ void see(IU w, int base, int trace=0);            ///< disassemble user defined word
    __HOST__ void dict_dump();                                 ///< dump dictionary
    __HOST__ void mem_stat();                                  ///< display memory statistics
    ///
    /// self tests
    ///
    __HOST__ void self_tests();

private:
    ///
    /// methods for supporting see
    ///
    __HOST__ char *_d2h(const char *d_str);                   ///< convert from device string
    __HOST__ int  _p2didx(Param *p);                          ///< reverse lookup
    __HOST__ int  _to_s(IU w, int base, int trace=0);         ///< show dictionary entry
    __HOST__ int  _to_s(Param *p, int nv, int base, int trace=0); ///< show by parameter memory pointer
};
///@}
#endif // __DEBUG_H
