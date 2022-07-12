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
    VLOG1("\\  VM[%d](mem=%p, vmss=%p)\n",
          blockIdx.x, mmu.pmem(0), mmu.vmss(blockIdx.x));
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
    VLOG1("%c< %s\n", compile ? ':' : '<', fin.rdbuf()); /// * display input buffer
    while (fin >> idiom) {                   /// * loop throught tib
        VLOG2("%d>> %-10s => ", blockIdx.x, idiom);
        if (!parse(idiom) && !number(idiom)) {
            fout << idiom << "? " << ENDL;   /// * display error prompt
            compile = false;                 /// * reset to interpreter mode
        }
    }
    if (!compile) ss_dump(ss.idx);
}
//=======================================================================================
