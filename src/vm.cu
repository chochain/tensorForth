/** -*- c++ -*-
 * @File
 * @brief - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mmu.h"
#include "vm.h"

__GPU__
VM::VM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0)
    : khz(khz), fin(*istr), fout(*ostr), mmu(*mmu0) {
    if (mmu0->trace() > 0) {
        printf("\\  VM[%d](mem=%p, vmss=%p)\n",
               blockIdx.x, mmu.pmem(0), mmu.vmss(blockIdx.x));
    }
}
///==============================================================================
///
/// debug methods
///
__GPU__ void
VM::dot(DU v) {
    if (IS_OBJ(v)) { fout << v; mmu.mark_free(v); }
    else fout << " " << v;       // eForth has a space prefix
}
__GPU__ void
VM::dot_r(int n, DU v) {
    fout << setw(n) << v;
}
__GPU__ void
VM::ss_dump(int n) {
    ss[T4_SS_SZ-1] = top;        // put top at the tail of ss (for host display)
    fout << opx(OP_SS, n);
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
    while (fin >> idiom) {                   /// loop throught tib
        WARN("%d>> %-10s => ", blockIdx.x, idiom);
        if (!parse(idiom) && !number(idiom)) {
            fout << idiom << "? " << ENDL;   ///> display error prompt
            compile = false;                 ///> reset to interpreter mode
        }
    }
    if (!compile) ss_dump(ss.idx);
}
//=======================================================================================
