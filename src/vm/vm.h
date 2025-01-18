/**
 * @file
 * @brief VM class - eForth VM virtual class interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_VM_H
#define TEN4_SRC_VM_H
#include "aio.h"            // async IO (includes Istream, Ostream), in ../io
///
///@name Cross platform support
///@{
#define ENDL         '\n'
#define delay(ticks) { U64 t = clock64() + (ticks * mmu.khz()); while ((U64)clock64()<t) yield(); }
#define yield()                        /**< TODO: multi-VM  */
///@}
#define VLOG1(...)         if (mmu.trace() > 0) INFO(__VA_ARGS__);
#define VLOG2(...)         if (mmu.trace() > 1) INFO(__VA_ARGS__);
///
/// virtual machine base class
///
typedef enum { VM_READY=0, VM_RUN, VM_WAIT, VM_STOP } vm_state;
class ALIGNAS VM {
public:    
    IU        vid    = 0;              ///< VM id
    vm_state  state  = VM_STOP;        ///< VM state
    
    DU        tos    = DU0;            ///< cached top of stack
    IU        wp     = 0;
    
    Vector<DU, 0> ss;                  ///< parameter stack (setup in ten4.cu)
    Vector<DU, 0> rs;                  ///< return stack    (setup in ten4.cu)

    cudaEvent_t   event;
    cudaStream_t  stream;

    static void vm_init(Istream *istr, Ostream *ostr, MMU *mmu) {
        fin  = *istr;
        fout = *ostr;
        mmu  = *mmu0;
    }
    __GPU__ VM(int id, Istream *istr, Ostream *ostr, MMU *mmu);

    __GPU__ virtual void init() { VLOG1("VM::init ok\n"); }
    __GPU__ virtual void outer();
    
#if DO_MULTITASK
    static int      NCORE;         ///< number of hardware cores
    
    static bool     io_busy;       ///< IO locking control
    static MUTEX    io;            ///< mutex for io access
    static MUTEX    tsk;           ///< mutex for tasker
    static COND_VAR cv_io;         ///< io control
    static COND_VAR cv_tsk;        ///< tasker control
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
    ///
    /// IO interface
    ///
    void io_lock();                   ///< lock IO
    void io_unlock();                 ///< unlock IO
#endif // DO_MULTITASK

protected:
    U32   *ptop   = (U32*)&top;       ///< 32-bit mask for top
    U8    *base   = 0;                ///< radix
    bool  compile = false;            ///< compiling flag
    char  idiom[T4_STRBUF_SZ];        ///< terminal input buffer
    
    static Istream  &fin;             ///< VM stream input
    static Ostream  &fout;            ///< VM stream output
    static MMU      &mmu;             ///< memory managing unit
    ///
    /// inner interpreter handlers
    ///
    __GPU__ virtual int resume()          { return 0; }
    __GPU__ virtual int pre(char *str)    { return 0; }
    __GPU__ virtual int parse(char *str)  { return 0; }
    __GPU__ virtual int number(char *str) { return 0; }
    __GPU__ virtual int post()            { return 0; }
    ///
    /// input stream handler
    ///
    __GPU__ char *next_idiom()      { fin >> idiom; return idiom; }
    __GPU__ char *scan(char delim)  { fin.get_idiom(idiom, delim); return idiom; }
    ///
    /// output methods
    ///
    __GPU__ void dot(DU v)          { fout << " " << v; }
    __GPU__ void dot_r(int n, DU v) { fout << setw(n) << v; }
    __GPU__ void ss_dump(int n=0)   {
        ss[T4_SS_SZ-1] = top;        /// * put top at the tail of ss (for host display)
        fout << opx(OP_SS, n ? n : ss.idx);
    }
};
#endif // TEN4_SRC_VM_H
