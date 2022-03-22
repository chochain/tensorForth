#ifndef __EFORTH_SRC_EFORTH_H
#define __EFORTH_SRC_EFORTH_H
#include "cuef_types.h"
#include "vector.h"         // cueForth vector
#include "strbuf.h"
//#include "sstream.h"		// cueForth sstream

#define ENDL            "\n"
#define millis()        clock()
#define delay(ms)       { clock_t t = clock()+ms; while (clock()<t); }
#define yield()

typedef GF    DTYPE;
#define DVAL  0.0f

class Code;                                 /// forward declaration
struct fop {                                /// alternate solution for function
    __GPU__ virtual void operator()(Code*) = 0;
};
template<typename F>
struct function : fop {
    F& fp;
    __GPU__ function(F& f) : fp(f) {}
    __GPU__ void operator()(Code *c) { fp(c); }
};

class Code {
public:
    const char *name = 0;   /// name field
    union {                 /// either a primitive or colon word
        fop *xt = 0;        /// lambda pointer
        struct {            /// a colon word
            U16 def:  1;    /// colon defined word
            U16 immd: 1;    /// immediate flag
            U16 len:  14;   /// len of pfa
            IU  pfa;        /// offset to pmem space
        };
    };
    template<typename F>    /// template function for lambda
    __GPU__ Code(const char *n, F f, bool im=false) : name(n) {
        xt = new function<F>(f);
        immd = im ? 1 : 0;
    }
    __GPU__ Code() {}       /// create a blank struct (for initilization)
};
///
/// Forth Virtual Machine operational macros
///
#define STRLEN(s) (ALIGN(d_strlen(s,0)+1))  /** calculate string size with alignment     */
#define XIP       (dict[-1].len)            /** parameter field tail of latest word      */
#define PFA(w)    ((U8*)&pmem[dict[w].pfa]) /** parameter field pointer of a word        */
#define PFLEN(w)  (dict[w].len)             /** parameter field length of a word         */
#define CELL(a)   (*(DU*)&pmem[a])          /** fetch a cell from parameter memory       */
#define STR(a)    ((char*)&pmem[a])         /** fetch string pointer to parameter memory */
#define JMPIP     (IP0 + *(IU*)IP)          /** branching target address                 */
#define SETJMP(a) (*(IU*)(PFA(-1) + (a)))   /** address offset for branching opcodes     */
#define HERE      (pmem.idx)                /** current parameter memory index           */
#define IPOFF     ((IU)(IP - PMEM0))        /** IP offset relative parameter memory root */
#define CALL(c)\
	if (dict[c].def) nest(c);\
    else (*(fop*)(((uintptr_t)dict[c].xt)&~0x3))(c)
///
/// Forth virtual machine class
///
class ForthVM {
public:
//  istream       &cin;                     /// stream input
//	ostream       &cout;				    /// stream output

    Vector<DU,   64>      rs;               /// return stack
    Vector<DU,   64>      ss;               /// parameter stack
    Vector<Code, 1024>    dict;				/// dictionary
    Vector<U8,   48*1024> pmem;             /// primitives

    bool  compile = false, ucase = true;    /// compiling flag
    int   base    = 10;                     /// numeric radix
    DU    top     = DVAL;                   /// cached top of stack
    IU    WP      = 0;                      /// instruction and parameter pointers
    U8    *PMEM0  = &pmem[0];
    U8    *IP = PMEM0, *IP0 = PMEM0;

 //   __GPU__ ForthVM(istream &in, ostream &out);

    __GPU__ void init();
    __GPU__ void outer();

private:
    __GPU__ DU   POP()        { DU n=top; top=ss.pop(); return n; }
    __GPU__ DU   PUSH(DU v)   { ss.push(top); top = v; }
    
    __GPU__ int  streq(const char *s1, const char *s2);
    __GPU__ int  find(const char *s);      /// search dictionary reversely
    ///
    /// Forth compiler functions
    ///
    __GPU__ void add_iu(IU i);             /// add an instruction into pmem
    __GPU__ void add_du(DU v);             /// add a cell into pmem
    __GPU__ void add_str(const char *s);   /// add a string to pmem
    __GPU__ void colon(const char *name);
    ///
    /// Forth inner interpreter
    ///
    __GPU__ char *next_word();
    __GPU__ char *scan();
    __GPU__ void nest(IU c);
    __GPU__ void call(Code *c);             /// execute a word
    ///
    /// debug functions
    ///
    __GPU__ void dot_r(int n, DU v);
    __GPU__ void to_s(IU c);
    __GPU__ void see(IU *cp, IU *ip, int dp=0);
    __GPU__ void words();
    __GPU__ void ss_dump();
    __GPU__ void mem_dump(IU p0, DU sz);
};
#endif // __EFORTH_SRC_EFORTH_H
