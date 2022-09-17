/** -*- c++ -*-
 * @File
 * @brief - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "vm.h"

__GPU__
VM::VM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0)
    : khz(khz), fin(*istr), fout(*ostr), mmu(*mmu0) {
    vid = threadIdx.x;
    VLOG1("\\  VM[%d](mem=%p, vmss=%p)\n", vid, mmu.pmem(0), mmu.vmss(vid));
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
    while (status == VM_READY && fin >> idiom) { /// * loop throught tib
        if (pre(idiom)) continue;                /// * pre process
        VLOG2("%d| >> %-10s => ", vid, idiom);
        if (!parse(idiom) && !number(idiom)) {
            fout << idiom << "? " << ENDL;       /// * display error prompt
            compile = false;                     /// * reset to interpreter mode
        }
        if (post()) break;                       /// * post process
    }
    if (status==VM_READY && !compile) ss_dump(ss.idx);
}
//=======================================================================================
