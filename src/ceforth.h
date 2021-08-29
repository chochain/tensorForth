#ifndef __EFORTH_SRC_CEFORTH_H
#define __EFORTH_SRC_CEFORTH_H
#include "vector.h"         // cueForth vector
#include "sstream.h"		// cueForth sstream
#include <functional>       // function
#include <exception>
#include <chrono>
#include <thread>
#include "cuef.h"

#define ENDL            "\n"
#define millis()        clock()
#define delay(ms)       { clock_t t = clock()+ms; while (clock()<t); }
#define yield()

typedef GF    DTYPE;
#define DVAL  0.0f

namespace cuef {

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
    string name;                            /// name of word
    int    token = 0;                       /// dictionary order token
    bool   immd  = false;                   /// immediate flag
    int    stage = 0;                       /// branching stage
    fop    *xt   = NULL;                    /// primitive function
    string literal;                         /// string literal

    vector<Code*> pf;
    vector<Code*> pf1;
    vector<Code*> pf2;
    vector<DTYPE> qf;

    template<typename F>
    __GPU__ Code(string n, F fn, bool im=false);	  /// primitive
    __GPU__ Code(string n, bool f=false);             /// new colon word or temp
    __GPU__ Code(Code *c,  DTYPE d);                  /// dolit, dovar
    __GPU__ Code(Code *c,  string s=string(""));      /// dotstr

    __GPU__ Code     *addcode(Code *w);               /// append colon word
    __GPU__ string&  to_s();                          /// debugging
    __GPU__ string&  see(int dp);
    __GPU__ void     nest();                          /// execute word
};
///
/// Forth virtual machine variables
///
class ForthVM {
public:
    istream       &cin;                     /// stream input
	ostream       &cout;				    /// stream output

    vector<DTYPE> rs;                       /// return stack
    vector<DTYPE> ss;                       /// parameter stack
    vector<Code*> dict;                     /// dictionary

    bool  compile = false;                  /// compiling flag
    int   base    = 10;                     /// numeric radix
    int   WP      = 0;                      /// instruction and parameter pointers
    DTYPE top     = DVAL;                   /// cached top of stack

    __GPU__ ForthVM(istream &in, ostream &out);

    __GPU__ void init();
    __GPU__ void outer();

private:
    __GPU__ DTYPE POP();
    __GPU__ DTYPE PUSH(DTYPE v);
    
    __GPU__ Code *find(const char *s);              /// search dictionary reversely
    __GPU__ Code *find(string &s);                  /// search dictionary reversely
    __GPU__ string& next_idiom(char delim=0);
    __GPU__ void call(Code *c);                     /// execute a word
    __GPU__ void call(vector<Code*> pf);
    
    __GPU__ void dot_r(int n, DTYPE v);
    __GPU__ void ss_dump();
    __GPU__ void words();
};

} // namespace cuef

extern __KERN__ void vm_pool_init(U8 *cin, U8 *cout);

#endif // __EFORTH_SRC_CEFORTH_H
