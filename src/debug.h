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
    __HOST__ ~Debug();
    
    __HOST__ int  to_s(DU s);                             ///< dump object from descriptor
    __HOST__ int  to_s(T4Base &t, bool view);             ///< dump object on stack
    
    __HOST__ void words(int rdx);                         ///< list dictionary words
    __HOST__ void see(IU pfa, int rdx);                   ///< disassemble user defined word
    __HOST__ void ss_dump(DU *ss, U16 n, int rdx);        ///< show data stack content
    __HOST__ void mem_dump(U32 addr, IU sz, int base);    ///< dump memory frm addr...addr+sz
    __HOST__ void dict_dump(int base);                    ///< dump dictionary
    __HOST__ void mem_stat();                             ///< display memory statistics
};
#endif // TEN4_SRC_DEBUG_H
