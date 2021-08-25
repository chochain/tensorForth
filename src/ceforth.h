#ifndef __EFORTH_SRC_CEFORTH_H
#define __EFORTH_SRC_CEFORTH_H
#include <sstream>
#include <vector>           // vector
#include <functional>       // function
#include <exception>
#include <chrono>
#include <thread>
#include "cuef.h"

#define ENDL endl
#define millis()        chrono::duration_cast<chrono::milliseconds>( \
							chrono::steady_clock::now().time_since_epoch()).count()
#define delay(ms)       this_thread::sleep_for(chrono::milliseconds(ms))
#define yield()         this_thread::yield()

typedef float DTYPE;
#define DVAL  0.0f

using namespace std;

template<class T>
struct ForthList {          /// vector helper template class
    T 		v[256];         /// use proxy pattern
    int     sz =0;

    __GPU__ T& operator[](int i) { return i < 0 ? v[sz + i] : v[i]; }
    __GPU__ T operator<<(T t)    { v[sz++] = t; }

    __GPU__ T dec_i() { return v[sz - 1] -= 1; }     /// decrement stack top
    __GPU__ T pop()   { return sz>0 ? v[(sz--)-1] : 0; }
    __GPU__ int  size()              { return sz; }
    __GPU__ void push(T t)           { v[sz++] = t; }
    __GPU__ void clear(int i=0)      { sz = i; }
    __GPU__ void merge(ForthList& a) {
    	for (int i=0; i<a.size(); i++) push(a[i]);
    }
    __GPU__ void merge(T a, int len) {
    	for (int i=0; i<len; i++) push(a++);
    }
};

class Code;                                 /// forward declaration
using fop = function<void(Code*)>;         /// Forth operator

class Code {
public:
    string name;                            /// name of word
    int    token = 0;                       /// dictionary order token
    bool   immd  = false;                   /// immediate flag
    int    stage = 0;                       /// branching stage
    fop    xt    = NULL;                    /// primitive function
    string literal;                         /// string literal

    ForthList<Code*> pf;
    ForthList<Code*> pf1;
    ForthList<Code*> pf2;
    ForthList<DTYPE> qf;

#if NO_FUNCTION
    template<typename F>
    __GPU__ Code(string n, F fn, bool im=false);	/// primitive
#else
    __GPU__ Code(string n, fop fn, bool im=false);  /// primitive
#endif // NO_FUNCTION
    __GPU__ Code(string n, bool f=false);           /// new colon word or temp
    __GPU__ Code(Code *c,  DTYPE d);                /// dolit, dovar
    __GPU__ Code(Code *c,  string s=string());      /// dotstr

    __GPU__ Code *addcode(Code *w);                 /// append colon word
    __GPU__ string to_s();                          /// debugging
    __GPU__ string see(int dp);
    __GPU__ void   nest();                          /// execute word
};
///
/// Forth virtual machine variables
///
class ForthVM {
public:
    istream          &cin;                  /// stream input
	ostream          &cout;					/// stream output

    ForthList<DTYPE> rs;                    /// return stack
    ForthList<DTYPE> ss;                    /// parameter stack
    ForthList<Code*> dict;                  /// dictionary

    bool  compile = false;                  /// compiling flag
    int   base    = 10;                     /// numeric radix
    int   WP      = 0;                      /// instruction and parameter pointers
    DTYPE top     = -1.0;                   /// cached top of stack

    __GPU__ ForthVM(istream &in, ostream &out);

    __GPU__ void init();
    __GPU__ void outer();

private:
    __GPU__ DTYPE POP();
    __GPU__ DTYPE PUSH(DTYPE v);
    
    __GPU__ Code *find(string s);                   /// search dictionary reversely
    __GPU__ string next_idiom(char delim=0);
    __GPU__ void call(Code *c);                     /// execute a word
    
    __GPU__ void dot_r(int n, DTYPE v);
    __GPU__ void ss_dump();
    __GPU__ void words();
};
#endif // __EFORTH_SRC_CEFORTH_H
