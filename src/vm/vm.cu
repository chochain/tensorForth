/** -*- c++ -*-
 * @file
 * @brief VM class - tensorForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vm.h"

__GPU__ 
VM::VM(int id, System *sys) 
    : id(id), state(STOP), sys(sys), mmu(sys->mu) {
    SS.init(mmu->vmss(id), T4_SS_SZ);
    RS.init(mmu->vmrs(id), T4_RS_SZ);
    TRACE("\\ VM[%d] created, sys=%p ss=%p, rs=%p\n", id, sys, SS.v, RS.v);
}
///
/// VM Outer interpreter
/// @brief having outer() on device creates branch divergence but
///    + can enable parallel VMs (with different tasks)
///    + can support parallel find()
///    + can support system without a host
///    However, to optimize,
///    + compilation can be done on host and
///    + only call() is dispatched to device
///    + number() and find() can run in parallel
///    - however, find() can run in serial only
///
__GPU__ void
VM::outer() {
    char *idiom;
    if (state == NEST) resume();                     /// * resume from suspended VM
    while ((idiom = sys->fetch())) {                 /// * loop throught tib
        if (pre(idiom)) continue;                    /// * pre process
        DEBUG("%d> idiom='%s' => ", id, idiom);
        if (!process(idiom)) {
            sys->perr(idiom, "? ");                  /// * display error prompt
            compile = false;                         /// * reset to interpreter mode
            state   = QUERY;                         /// * back to input mode
            break;                                   /// * bail
        }
        if (post()) break;                           /// * post process
    }
    TRACE("%d> VM.state=%d\n", id, state);
}
//=======================================================================================
