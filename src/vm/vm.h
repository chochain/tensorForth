/**
 * @file
 * @brief VM class - eForth VM virtual class interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_VM_H
#define TEN4_SRC_VM_H
#include "sys.h"                    /// system interface
///
///@name Cross platform support
///@{
#define yield()            /**< TODO: multi-VM  */
#define delay(ticks) {                            \
        U64 t = clock64() + (ticks * sys->khz()); \
        while ((U64)clock64()<t) yield();         \
}
///@}
#define VLOG1(...) if (sys->trace() > 0) INFO(__VA_ARGS__);
#define VLOG2(...) if (sys->trace() > 1) INFO(__VA_ARGS__);
///
/// virtual machine base class
///
typedef enum { STOP=0, HOLD, QUERY, NEST } vm_state;   ///< ten4 states
class VM {                            ///< VM (created in kernal mode)
public:
    IU        id;                     ///< VM id
    vm_state  state;                  ///< VM state
    System    *sys;                   ///< system interface
    MMU       *mmu;                   ///< cached MMU interface
    
    Vector<DU, 0> ss;                 ///< parameter stack (setup in ten4.cu)
    Vector<DU, 0> rs;                 ///< return stack

    __GPU__  VM(int id, System *sys);
    __GPU__  ~VM() { VLOG1("%d ", id); }
    
    __GPU__  virtual void    init() { VLOG1("VM[%d]::init ok\n", id); }
    __GPU__  virtual void    outer();
    ///
    /// proxy methods to MMU
    ///
    __GPU__ __INLINE__ DU    DUP(DU d)  { return IS_OBJ(d) ? AS_VIEW(d) : d; }  ///< soft copy
#if T4_ENABLE_OBJ        
    __GPU__ __INLINE__ DU    COPY(DU d) {                                       ///< hard copy
        return (IS_OBJ(d))
            ? T4Base::obj2du(mmu->copy((Tensor&)T4Base::du2obj(d)))
            : d;
    }
    __GPU__ __INLINE__ void  DROP(DU d) { mmu->drop(t); }                       ///< free obj
#else  // !T4_ENABLE_OBJ    
    __GPU__ __INLINE__ DU    COPY(DU d) { return d; }
    __GPU__ __INLINE__ void  DROP(DU d) {}
#endif // T4_ENABLE_OBJ
    
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
#endif // DO_MULTITASK

protected:
    bool  compile = false;            ///< compiling flag
    ///
    /// inner interpreter handlers
    ///
    __GPU__ virtual int resume()          { return 0; }
    __GPU__ virtual int pre(char *str)    { return 0; }
    __GPU__ virtual int parse(char *str)  { return 0; }
    __GPU__ virtual int number(char *str) { return 0; }
    __GPU__ virtual int post()            { return 0; }
};
#endif // TEN4_SRC_VM_H
