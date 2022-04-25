/*! @file
  @brief
  cueForth - eForth core classes
*/
#ifndef CUEF_SRC_EFORTH_H
#define CUEF_SRC_EFORTH_H
#include "cuef_types.h"
#include "util.h"
#include "vector.h"         // cueForth vector
#include "aio.h"            // cueForth async IO (Istream, Ostream)

#define ENDL            "\n"
#define millis()        clock()
#define delay(ms)       { clock_t t = clock()+ms; while (clock()<t); }
#define yield()
///
/// Forth Virtual Machine operational macros
///
//#define STR(a)    ((char*)&_pmem[a])        /** fetch string pointer to parameter memory */

#define INT(f)    (static_cast<int>(f))       /** cast float to int                        */
#define I2DU(i)   (static_cast<DU>(i))        /** cast int back to float                   */
#define LWIP      (dict[-1]->len)             /** parameter field tail of latest word      */
#define JMPIP     (IP0 + *(IU*)IP)            /** branching target address                 */
#define IPOFF     ((IU)(IP - PMEM0))          /** IP offset relative parameter memory root */
#define FIND(s)   (dict.find(s, compile, ucase))
#define CALL(c)\
    if (dict[c]->def) nest(c);\
    else (*(fop*)(((uintptr_t)dict[c]->xt)&~0x3))(c)
///
/// Forth virtual machine class
///
typedef enum { VM_READY=0, VM_RUN, VM_WAIT, VM_STOP } vm_status;

class Dict;
class ForthVM {
public:
    Istream       &fin;                     /// VM stream input
    Ostream       &fout;                    /// VM stream output
    Dict          &dict;                    /// dictionary object
    vm_status     status = VM_READY;        /// VM status

    Vector<DU,   CUEF_RS_SZ>   rs;          /// return stack
    Vector<DU,   CUEF_SS_SZ>   ss;          /// parameter stack

    bool  compile = false;                  /// compiling flag
    bool  ucase   = true;                   /// case insensitive
    int   base    = 10;                     /// numeric radix
    DU    top     = DU0;                    /// cached top of stack
    IU    WP      = 0;                      /// word and parameter pointers
    U8    *PMEM0, *IP0, *IP;                /// cached base-memory pointer

    char  idiom[80];                        /// terminal input buffer

    __GPU__ ForthVM(
        Istream *istr,
        Ostream *ostr,
        Dict    *dict0)
    : fin(*istr), fout(*ostr), dict(*dict0) {
    	printf("dict=%p\n", dict0);
    	//PMEM0 = IP0 = IP = dict0;
    }

    __GPU__ void init();
    __GPU__ void outer();

private:
    __GPU__ DU   POP()        { DU n=top; top=ss.pop(); return n; }
    __GPU__ DU   PUSH(DU v)   { ss.push(top); top = v; }

    __GPU__ int  find(const char *s);      /// search dictionary reversely
    ///
    /// Forth inner interpreter
    ///
    __GPU__ char *next_word();
    __GPU__ char *scan(char c);
    __GPU__ void nest(IU c);
    ///
    /// debug functions
    ///
    __GPU__ void dot_r(int n, DU v);
    __GPU__ void ss_dump();
};
#endif // CUEF_SRC_EFORTH_H
