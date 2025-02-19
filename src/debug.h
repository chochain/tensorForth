/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_DEBUG_H
#define TEN4_SRC_DEBUG_H
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
    MMU      *mu;                                         ///< memory management unit
    AIO      *io;                                         ///< async IO unit
    
public:
    __HOST__ Debug(MMU *mmu, AIO *aio) : mu(mmu), io(aio) {}
    __HOST__ ~Debug() {}

    __HOST__ void keep_fmt();
    __HOST__ void reset_fmt();
    
    __HOST__ void ss_dump(IU id, int sz, DU tos, int base=10); ///< show data stack content
    __HOST__ int  p2didx(Param *p);                       ///< reverse lookup
    __HOST__ int  to_s(IU w, int base=10);                ///< show dictionary info from descriptor
    __HOST__ int  to_s(Param *p, int nv, int base);
    __HOST__ void words();                                ///< list dictionary words
    __HOST__ void mem_dump(IU addr, int sz);              ///< dump memory frm addr...addr+sz
    __HOST__ void see(IU w, int base=10);                 ///< disassemble user defined word
    __HOST__ void dict_dump();                            ///< dump dictionary
    __HOST__ void mem_stat();                             ///< display memory statistics
    ///
    /// self tests
    ///
    __HOST__ void self_tests();
};
///@}
#endif // TEN4_SRC_DEBUG_H
