/** -*- c++ -*-
 * @file
 * @brief VM class - tensorForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "t4math.h"
#include "vm.h"
#include "eforth.h"
#include "tenvm.h"
#include "netvm.h"

namespace t4::vm {

VM *vm_factory(vm_level level, int id, System &sys) {
    switch (level) {
    case NET   : return new vm::NetVM(id, sys);
    case TENSOR: return new vm::TensorVM(id, sys);
    default    : return new vm::ForthVM(id, sys);
    }
}

__HOST__ 
VM::VM(int id, System &sys) 
    : id(id), state(STOP), sys(sys), mmu(*sys.mu) {
    ss.init(mmu.vmss(id), T4_SS_SZ);
    rs.init(mmu.vmrs(id), T4_RS_SZ);
    TRACE("\\ VM[%d] created, sys=%p ss=%p, rs=%p\n", id, &sys, &ss[0], &rs[0]);
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
__HOST__ void
VM::outer() {
    char *idiom;
    while ((idiom = sys.fetch())!=0) {               /// * loop throught tib
        DEBUG("vm%d> idiom='%s' => ", id, idiom);
        if (pre(idiom)) continue;                    /// * pre process (filter)
        if (!process(idiom)) {
            sys.perr(idiom, "? ");                   /// * display error prompt
            sys.clrbuf();                            /// * flush input stream
            compile = false;                         /// * reset to interpreter mode
            state   = QUERY;                         /// * back to input mode
            break;                                   /// * bail
        }
        if (state==HOLD) break;
    }
    post();                                          /// * post process (debug)
}
///
///@name ALU opcodes (1-operand and 2-operand)
///@{
__HOST__ void
VM::xop1(math_op op, DU v) {                         ///< single operand operator
    DU t = tos;
    switch (op) {
    case ABS:  t = ABS(t);          break;
    case NEG:  t = NEG(t);          break;
    case EXP:  t = EXP(t);          break;
    case LN:   t = t > DU_EPS ? LN(t)  : DU0;  break;
    case LOG:  t = t > DU_EPS ? LOG(t) : DU0;  break;
    case TANH: t = TANH(t);         break;
    case RELU: t = MAX(t, DU0);     break;
    case SIGM: t = SIGMOID(t);      break;
    case SQRT: t = SQRT(t);         break;
    case RCP:  t = RCP(t);          break;
    case SAT:  t = SAT(t);          break;
    case SIN:  t = SIN(t);          break;
    case COS:  t = COS(t);          break;
    default: NA("op=%d?\n");        break;
    }
    tos = SCALAR(t);
}

__HOST__ void
VM::xop2(math_op op, t4_drop_opt x) {               ///< 2-operand operator
    DU t = tos, n = ss.pop();
    switch (op) {
    case ADD:  t = ADD(n, t);       break;
    case MUL:  t = MUL(n, t);       break;
    case SUB:  t = SUB(n, t);       break;
    case DIV:  t = DIV(n, t);       break;
    case MOD:  t = MOD(n, t);       break;
    case MAX:  t = MAX(n, t);       break;
    case MIN:  t = MIN(n, t);       break;
    case MUL2: t = MUL2(n,t);       break;
    case MOD2: t = MOD2(n,t);       break;
    case POW:  t = POW(t, n);       break;
    default: NA("op=%d?\n");        break;
    }
    tos = SCALAR(t);
}

} // namespace t4::vm
//=======================================================================================
