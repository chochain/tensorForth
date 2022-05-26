/*! @file
  @brief
  tensorForth - eForth core classes
*/
#ifndef TEN4_SRC_EFORTH_H
#define TEN4_SRC_EFORTH_H
#include "ten4_types.h"
#include "util.h"
#include "vector.h"         // Forth vector
#include "aio.h"            // Forth async IO (Istream, Ostream)

#define ENDL            '\n'
#define millis()        (clock()/1530000.0f)
#define delay(ms)       { clock_t t = clock()+ms; while (clock()<t); }
#define yield()
///
/// Forth virtual machine class
///
typedef enum { VM_READY=0, VM_RUN, VM_WAIT, VM_STOP } vm_status;

class Dict;
class ForthVM {
public:
    vm_status     status = VM_READY;        /// VM status
    DU    top    = DU0;                     /// cached top of stack
    Vector<DU,   T4_RS_SZ> rs;              /// return stack
    Vector<DU,   0>        ss;              /// parameter stack (setup in ten4.cu)

    __GPU__ ForthVM(Istream *istr, Ostream *ostr, MMU *mmu);
    __GPU__ void init();
    __GPU__ void outer();

private:
    Istream       &fin;                     /// VM stream input
    Ostream       &fout;                    /// VM stream output
    MMU           &mmu;                     /// memory managing unit
    Code          *dict;                    /// dictionary array

    bool  ucase   = true;                   /// case insensitive
    bool  compile = false;                  /// compiling flag
    int   radix   = 10;                     /// numeric radix
    IU    WP      = 0;                      /// word and parameter pointers
    IU    IP      = 0;                      /// instruction pointer
    IU    NXT;                              /// cached DONEXT xt address, used in nest()

    char  idiom[T4_STRBUF_SZ];              /// terminal input buffer

    __GPU__ __INLINE__ DU POP()        { DU n=top; top=ss.pop(); return n; }
    __GPU__ __INLINE__ void PUSH(DU v) { ss.push(top); top = v; }

    __GPU__ int  find(const char *s);       /// search dictionary reversely
    ///
    /// Forth inner interpreter
    ///
    __GPU__ char *next_idiom();
    __GPU__ char *scan(char c);
    __GPU__ void nest();
    ///
    /// compiler proxy funtions to reduce verbosity
    ///
    __GPU__ void add_iu(IU i);              /// append an instruction unit to parameter memory
    __GPU__ void add_du(DU d);              /// append a data unit to pmem
    __GPU__ void add_str(const char *s);    /// append a string to pmem
    __GPU__ void add_w(IU w);               /// append a word pfa to pmem
    __GPU__ void call(IU w);
    ///
    /// debug functions
    ///
    __GPU__ void dot_r(int n, DU v);
    __GPU__ void ss_dump(int n);
};
#endif // TEN4_SRC_EFORTH_H
