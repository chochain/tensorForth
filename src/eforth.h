#ifndef __EFORTH_SRC_EFORTH_H
#define __EFORTH_SRC_EFORTH_H
#include "cuef_types.h"
#include "vector.h"         // cueForth vector
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
    const char   *name;                     /// name of word
    int    token = 0;                       /// dictionary order token
    bool   immd  = false;                   /// immediate flag
    int    stage = 0;                       /// branching stage
    fop    *xt   = NULL;                    /// primitive function
    const char   *literal;                  /// StrBuf literal

    Vector<Code*> pf;
    Vector<Code*> pf1;
    Vector<Code*> pf2;
    Vector<DTYPE> qf;

    template<typename F>
    __GPU__ Code(const char *n, F fn, bool im=false); /// primitive
    __GPU__ Code(const char *n, bool f=false);        /// new colon word or temp
    __GPU__ Code(Code *c,  DTYPE d);                  /// dolit, dovar
    __GPU__ Code(Code *c,  const char *s="");         /// dotstr

    __GPU__ Code     *addcode(Code *w);               /// append colon word
    __GPU__ char     &to_s();                         /// debugging
    __GPU__ char     &see(int dp);
    __GPU__ void     nest();                          /// execute word
};
///
/// Forth virtual machine variables
///
class ForthVM {
public:
//  istream       &cin;                     /// stream input
//	ostream       &cout;				    /// stream output

    Vector<DTYPE> rs;                       /// return stack
    Vector<DTYPE> ss;                       /// parameter stack
    Vector<Code*> prim;                     /// primitives
    Vector<Code*> dict;						/// dictionary

    bool  compile = false;                  /// compiling flag
    int   base    = 10;                     /// numeric radix
    int   WP      = 0;                      /// instruction and parameter pointers
    DTYPE top     = DVAL;                   /// cached top of stack

 //   __GPU__ ForthVM(istream &in, ostream &out);

    __GPU__ void init();
    __GPU__ void outer();

private:
    __GPU__ DTYPE POP();
    __GPU__ DTYPE PUSH(DTYPE v);
    
    __GPU__ Code *find(const char *s);              /// search dictionary reversely
    __GPU__ char *next_idiom(char delim=0);
    __GPU__ void call(Code *c);                     /// execute a word
    __GPU__ void call(Vector<Code*> pf);
    
    __GPU__ void dot_r(int n, DTYPE v);
    __GPU__ void ss_dump();
    __GPU__ void words();
};
#endif // __EFORTH_SRC_EFORTH_H
