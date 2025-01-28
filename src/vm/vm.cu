/** -*- c++ -*-
 * @file
 * @brief VM class - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vm.h"

__HOST__
VM::VM(int id, System *sys)
    : id(id), state(STOP), sys(sys) {
    ss.init(sys->mu->vmss(id), T4_SS_SZ);      /// * point data stack to managed memory block
    VLOG1("\\  VM[%d](mem=%p, vmss=%p)\n", id, sys->mu->pmem(0), ss.v);
}
///
/// ForthVM Outer interpreter
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
    VLOG1("%d%c %s\n", id, compile ? ':' : '{', sys->io->fin.rdbuf()); /// * display input buffer
    char *idiom = sys->_pad;
    if (state == NEST) resume();                     /// * resume from suspended VM
    while (state == HOLD && sys->io->fin >> idiom) { /// * loop throught tib
        if (pre(idiom)) continue;                    /// * pre process
        VLOG2("%d| >> %-10s => ", id, idiom);
        if (!parse(idiom) && !number(idiom)) {
            sys->io->fout << idiom << "? " << ENDL;  /// * display error prompt
            compile = false;                         /// * reset to interpreter mode
        }
        if (post()) break;                           /// * post process
    }
}
//=======================================================================================
