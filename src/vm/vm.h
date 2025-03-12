/**
 * @file
 * @brief VM class - eForth VM virtual class interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VM_VM_H
#define __VM_VM_H
#include "sys.h"                    /// system interface
///
///@name virtual machine base class
///@{
typedef enum { STOP=0, HOLD, QUERY, NEST, VM_STATE_SZ } vm_state;   ///< ten4 states
class VM {                            ///< VM (created in kernal mode)
public:
    IU        id;                     ///< VM id
    vm_state  state   = STOP;         ///< VM state
    
    System    &sys;                   ///< system interface
    MMU       &mmu;                   ///< cached MMU interface

    Vector<DU, 0> ss;                 ///< parameter stack (setup in ten4.cu)
    Vector<DU, 0> rs;                 ///< return stack

    __GPU__  VM(int id, System &sys);
    __GPU__  ~VM() { TRACE("%d ", id); }
    
    __GPU__  virtual void   init()   { TRACE("VM[%d]::init ok\n", id); }
    __GPU__  virtual void   resume() {}
    __GPU__  virtual void   outer();
    
protected:
    bool  compile = false;            ///< compiling flag
    ///
    /// inner interpreter handlers
    ///
    __GPU__ virtual int pre(char *str)     { return 0; }
    __GPU__ virtual int process(char *str) { return 0; }
    __GPU__ virtual int post()             { return 0; }
    
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
#endif // __VM_VM_H
