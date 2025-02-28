/** 
 * @file
 * @brief Param class - tensorForth Parameter field definition
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VM_PARAM_H
#define __VM_PARAM_H
#include "ten4_types.h"
///
///@name Parameter Structure
///@{
struct Param : public Managed {
    union {
        IU pack;                   ///< collective
        struct {
            U32 ioff : 24;         ///< pfa, xtoff, or short int
            U32 op   : 4;          ///< opcode (1111 = colon word or built-in)
            U32 udf  : 1;          ///< user defined word
            U32 xx1  : 2;          ///< reserved
            U32 exit : 1;          ///< word exit flag
        };
    };
    __BOTH__ Param(prim_op o, IU ix, bool u=false, bool x=false) : pack(ix) {
        op=o; udf=u; exit=x;
    }
};
///@}
#endif  // __VM_PARAM_H
