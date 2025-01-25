/** -*- c++ -*-
 * @file
 * @brief VM class - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vm.h"

__GPU__
VM::VM(int id, System *sys)
    : vid(id), state(STOP), sys(*sys) {
    ss.init(sys.mmu.vmss(vid), T4_SS_SZ);      /// * point data stack to managed memory block
    VLOG1("\\  VM[%d](mem=%p, vmss=%p)\n", vid, sys.mmu.pmem(0), ss.v);
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
    VLOG1("%d%c %s\n", vid, compile ? ':' : '{', fin.rdbuf()); /// * display input buffer
    if (state == VM_RUN) resume();                 /// * resume from suspended VM
    while (state == VM_READY && fin >> idiom) {    /// * loop throught tib
        if (pre(idiom)) continue;                  /// * pre process
        VLOG2("%d| >> %-10s => ", vid, idiom);
        if (!parse(idiom) && !number(idiom)) {
            fout << idiom << "? " << ENDL;         /// * display error prompt
            compile = false;                       /// * reset to interpreter mode
        }
        if (post()) break;                         /// * post process
    }
}
//=======================================================================================
