/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_DEBUG_H
#define TEN4_SRC_DEBUG_H
#include "io/aio.h"
#include "mmu/mmu.h"
#include "mmu/t4base.h"                                   ///< include ten4_types.h

class Debug {                                             ///< friend class to MMU and AIO
    MMU *mu;                                              ///< memory management unit
    AIO *io;                                              ///< async IO unit
    
public:
    __HOST__ Debug(MMU *mmu, AIO *aio) : mu(mmu), io(aio) {}
    __HOST__ ~Debug() {}
    
    __HOST__ int  to_s(IU w);                             ///< show dictionary info from descriptor
    __HOST__ void words(int rdx=10);                      ///< list dictionary words
    __HOST__ void see(U8 *ip, int dp, int rdx=10);        ///< disassemble user defined word
    __HOST__ void see(IU w, int rdx=10);                  ///< disassemble user defined word
    __HOST__ void mem_dump(U32 addr, IU sz, int rdx=10);  ///< dump memory frm addr...addr+sz
    __HOST__ void dict_dump(int rdx=10);                  ///< dump dictionary
    __HOST__ void mem_stat();                             ///< display memory statistics
    __HOST__ void ss_dump(IU vid, U16 n, int rdx=10);     ///< show data stack content
};
#endif // TEN4_SRC_DEBUG_H
