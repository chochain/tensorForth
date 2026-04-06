/**
 * @file
 * @brief VM class - eForth VM virtual class interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VM_VM_H
#define __VM_VM_H
#pragma once

#include "sys.h"                      /// system interface

namespace t4::vm {
///
///@name ALU opcodes (1-operand and 2-operand)
///@{
#define XOP1(op, v)                             \
    DU t = tos;                                 \
    switch (op) {                               \
    case ABS:  t = ABS(t);          break;      \
    case NEG:  t = NEG(t);          break;      \
    case EXP:  t = EXP(t);          break;      \
    case LN:   t = LN(t);           break;      \
    case LOG:  t = LOG(t);          break;      \
    case TANH: t = TANH(t);         break;      \
    case RELU: t = MAX(t, DU0);     break;      \
    case SIGM: t = SIGMOID(t);      break;      \
    case SQRT: t = SQRT(t);         break;      \
    case RCP:  t = RCP(t);          break;      \
    case SAT:  t = SAT(t);          break;      \
    case POW:  t = POW(t, v);       break;      \
    }                                           \
    SCALAR(t); tos = t

#define XOP2(op)                                \
    DU t = tos, n = ss.pop();                   \
    switch (op) {                               \
    case ADD:  t = ADD(n, t);       break;      \
    case MUL:  t = MUL(n, t);       break;      \
    case SUB:  t = SUB(n, t);       break;      \
    case DIV:  t = DIV(n, t);       break;      \
    case MOD:  t = MOD(n, t);       break;      \
    case MAX:  t = MAX(n, t);       break;      \
    case MIN:  t = MIN(n, t);       break;      \
    case MUL2: t = MUL2(n,t);       break;      \
    case MOD2: t = MOD2(n,t);       break;      \
    }                                           \
    SCALAR(t); tos = t

///@}
///@name virtual machine base class
///@{
typedef enum { STOP=0, HOLD, QUERY, NEST, VM_STATE_SZ } vm_state;   ///< ten4 states

class VM {                            ///< VM (created in kernal mode)
public:
    IU        id;                     ///< VM id
    vm_state  state = STOP;           ///< VM state
    
    System    &sys;                   ///< system interface
    mu::MMU   &mmu;                   ///< cached MMU interface

    mu::Vector<DU, 0> ss;             ///< parameter stack (setup in ten4.cu)
    mu::Vector<DU, 0> rs;             ///< return stack
    
    IU    ip     = 0;                 ///< instruction pointer
    DU    tos    = -DU1;              ///< cached top of stack

    __HOST__  VM(int id, System &sys);
    __HOST__  ~VM() { TRACE("%d ", id); }
    ///
    /// VM life-cycle controls
    ///
    __HOST__  virtual void   init()   { TRACE("VM[%d]::init ok\n", id); }
    __HOST__  virtual void   resume() {}
    __HOST__  virtual void   outer();
    ///
    /// ALU operators
    ///
    __HOST__ virtual void xop1(math_op op, DU v=DU0) { XOP1(op, v); }           ///< single operand operator
    __HOST__ virtual void xop2(math_op op, t4_drop_opt x=T_KEEP) { XOP2(op); }  ///< 2-operand operator
    
protected:
    bool  compile = false;            ///< compiling flag
    ///
    /// inner interpreter handlers
    ///
    __HOST__ virtual int pre(char *str)     { return 0; }
    __HOST__ virtual int process(char *str) { return 0; }
    __HOST__ virtual int post()             { return 0; }

#if DO_MULTITASK    
    static MUTEX    tsk;              ///< mutex for tasker
    static COND_VAR cv_tsk;           ///< tasker control
    
    static void _ss_dup(VM &dst, VM &src, int n);
    ///
    /// task life cycle methods
    ///
    void reset(IU w, vm_state st);    ///< reset a VM user variables
    void join(int tid);               ///< wait for the given task to end
    void stop();                      ///< stop VM
    ///
    /// messaging interface
    ///
    void send(int tid, int n);        ///< send onto destination VM's stack (blocking, wait for receiver availabe)
    void recv();                      ///< receive data from any sending VM's stack (blocking, wait for sender's message)
    void bcast(int n);                ///< broadcast to all receivers
    void pull(int tid, int n);        ///< pull n items from the stack of a stopped task
#else  // DO_MULTITASK
    
    void set_state(vm_state st) { state = st; }
#endif // DO_MULTITASK    
};
///@}

} // namespace t4::vm
#endif // __VM_VM_H
