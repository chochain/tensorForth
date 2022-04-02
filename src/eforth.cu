/*! @file
  @brief
  cueForth Forth Vritual Machine implementation
*/
#include "eforth.h"
///
/// dictionary search functions - can be adapted for ROM+RAM
///
__GPU__ __INLINE__ int
ForthVM::streq(const char *s1, const char *s2) {
    return ucase ? STRCASECMP(s1, s2)==0 : STRCMP(s1, s2)==0;
}
__GPU__ int
ForthVM::find(const char *s) {
    for (int i = dict.idx - (compile ? 2 : 1); i >= 0; --i) {
        if (streq(s, dict[i].name)) return i;
    }
    return -1;
}
///==============================================================================
///
/// colon word compiler
/// Note:
///   * we separate dict and pmem space to make word uniform in size
///   * if they are combined then can behaves similar to classic Forth
///   * with an addition link field added.
///
enum {
    NOP = 0, DOVAR, DOLIT, DOSTR, DOTSTR, BRAN, ZBRAN, DONEXT, DOES, TOR
} forth_opcode;
///
/// Forth compiler functions
///
__GPU__ void                                /// add an instruction into pmem
ForthVM::add_iu(IU i) {
    pmem.push((U8*)&i, sizeof(IU));  XIP += sizeof(IU);
}
__GPU__ void                                /// add a cell into pmem
ForthVM::add_du(DU v) {
    pmem.push((U8*)&v, sizeof(DU)),  XIP += sizeof(DU);
}
__GPU__ void
ForthVM::add_str(const char *s) {           /// add a string to pmem
    int sz = STRASZ(s);
    pmem.push((U8*)s,  sz); XIP += sz;
}
__GPU__ void
ForthVM::colon(const char *name) {
    char *nfa = STR(HERE);                  // current pmem pointer
    int sz = STRASZ(name);                  // string length, aligned
    pmem.push((U8*)name,  sz);              // setup raw name field
    Code c(nfa, [](IU){});                  // create a new word on dictionary
    c.def = 1;                              // specify a colon word
    c.len = 0;                              // advance counter (by number of U16)
    c.pfa = HERE;                           // capture code field index
    dict.push(c);                           // deep copy Code struct into dictionary
};
///
/// Forth inner interpreter (colon word handler)
///
__GPU__ char*
ForthVM::next_word()  {     // get next idiom
    fin >> idiom; return idiom;
}
__GPU__ char*
ForthVM::scan(char c) {
    fin.get_idiom(idiom, c); return idiom;
}
__GPU__ void
ForthVM::nest(IU c) {
    rs.push(IP - PMEM0); rs.push(WP);       /// * setup call frame
    IP0 = IP = PFA(WP=c);                   // CC: this takes 30ms/1K, need work
//  try                                     // kernal does not support exception
    {                                       // CC: is dict[c] kept in cache?
        U8 *ipx = IP + PFLEN(c);            // CC: this saves 350ms/1M
        while (IP < ipx) {                  /// * recursively call all children
            IU c1 = *IP; IP += sizeof(IU);  // CC: cost of (ipx, c1) on stack?
            CALL(c1);                       ///> execute child word
        }                                   ///> can do IP++ if pmem unit is 16-bit
    }
//    catch(...) {}                         ///> protect if any exeception
    yield();                                ///> give other tasks some time
    IP0 = PFA(WP=rs.pop());                 /// * restore call frame
    IP  = PMEM0 + INT(rs.pop());
}
///==============================================================================
///
/// debug functions
///
__GPU__ void
ForthVM::dot_r(int n, DU v) {
    fout << setw(n) << setfill(' ') << v;
}
__GPU__ void
ForthVM::to_s(IU c) {
    fout << dict[c].name << " " << c << (dict[c].immd ? "* " : " ");
}
///
/// recursively disassemble colon word
///
__GPU__ void
ForthVM::see(IU *cp, IU *ip, int dp) {
    fout << ENDL; for (int i=dp; i>0; i--) fout << "  ";            // indentation
    if (dp) fout << "[" << setw(2) << *ip << ": ";                  // ip offset
    else    fout << "[ ";
    IU c = *cp;
    to_s(c);                                                        // name field
    if (dict[c].def) {                                              // a colon word
        for (IU n=dict[c].len, ip1=0; ip1<n; ip1+=sizeof(IU)) {     // walk through children
            IU *cp1 = (IU*)(PFA(c) + ip1);                          // next children node
            see(cp1, &ip1, dp+1);                                   // dive recursively
        }
    }
    switch (c) {
    case DOVAR: case DOLIT:
        fout << "= " << *(DU*)(cp+1); *ip += sizeof(DU); break;
    case DOSTR: case DOTSTR:
        fout << "= \"" << (char*)(cp+1) << '"';
        *ip += STRASZ((char*)(cp+1)); break;
    case BRAN: case ZBRAN: case DONEXT:
        fout << "j" << *(cp+1); *ip += sizeof(IU); break;
    }
    fout << "] ";
}
__GPU__ void
ForthVM::words() {
    fout << setbase(16);
    for (int i=0; i<dict.idx; i++) {
        if ((i%10)==0) { fout << ENDL; yield(); }
        to_s(i);
    }
    fout << setbase(base);
}
__GPU__ void
ForthVM::ss_dump() {
    fout << " <"; for (int i=0; i<ss.idx; i++) { fout << ss[i] << " "; }
    fout << top << "> ok" << ENDL;
}
__GPU__ void
ForthVM::mem_dump(IU p0, int sz) {
    fout << setbase(16) << setfill('0') << ENDL;
    for (IU i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        fout << setw(4) << i << ": ";
        for (int j=0; j<16; j++) {
            U8 c = pmem[i+j];
            fout << setw(2) << (int)c << (j%4==3 ? "  " : " ");
        }
        for (int j=0; j<16; j++) {   // print and advance to next byte
            U8 c = pmem[i+j] & 0x7f;
            fout << (char)((c==0x7f||c<0x20) ? '_' : c);
        }
        fout << ENDL;
        yield();
    }
    fout << setbase(base);
}
///================================================================================
///
/// macros to reduce verbosity
///
#define CODE(s, g)    { s, [this] __GPU__ (IU c){ g; }}
#define IMMD(s, g)    { s, [this] __GPU__ (IU c){ g; }, true }
#define BOOL(f)       ((f)?-1:0)
#define ALU(a, OP, b) (INT(a) OP INT(b))
///
/// global memory access macros
///
#define PEEK(a)        (U8)(*(U8*)((uintptr_t)(a)))
#define POKE(a, c)     (*(U8*)((uintptr_t)(a))=(U8)(c))
///
/// dictionary initializer
///
__GPU__ void
ForthVM::init() {
	const Code prim[] = {       /// singleton, build once only
    ///
    /// @defgroup Execution flow ops
    /// @brief - DO NOT change the sequence here (see forth_opcode enum)
    /// @{
    CODE("nop",     {}),
    CODE("dovar",   PUSH(IPOFF); IP += sizeof(DU)),
    CODE("dolit",   PUSH(*(DU*)IP); IP += sizeof(DU)),
    CODE("dostr",
        char *s = (char*)IP;                      // get string pointer
        PUSH(IPOFF); IP += STRASZ(s)),
    CODE("dotstr",
        char *s = (char*)IP;                      // get string pointer
        fout << s;  IP += STRASZ(s)),             // send to output console
    CODE("branch" , IP = JMPIP),                           // unconditional branch
    CODE("0branch", IP = POP() ? IP + sizeof(IU) : JMPIP), // conditional branch
    CODE("donext",
         if ((rs[-1] -= 1) >= 0) IP = JMPIP;       // rs[-1]-=1 saved 200ms/1M cycles
         else { IP += sizeof(IU); rs.pop(); }),
    CODE("does",                                   // CREATE...DOES... meta-program
         U8 *ip  = PFA(WP);
         U8 *ipx = ip + PFLEN(WP);                 // range check
         while (ip < ipx && *(IU*)ip != DOES) ip+=sizeof(IU);  // find DOES
         while ((ip += sizeof(IU)) < ipx) add_iu(*(IU*)ip);    // copy&paste code
         IP = ipx),                                            // done
    CODE(">r",   rs.push(POP())),
    CODE("r>",   PUSH(rs.pop())),
    CODE("r@",   PUSH(rs[-1])),
    /// @}
    /// @defgroup Stack ops
    /// @brief - opcode sequence can be changed below this line
    /// @{
    CODE("dup",  PUSH(top)),
    CODE("drop", top = ss.pop()),
    CODE("over", PUSH(ss[-1])),
    CODE("swap", DU n = ss.pop(); PUSH(n)),
    CODE("rot",  DU n = ss.pop(); DU m = ss.pop(); ss.push(n); PUSH(m)),
    CODE("pick", DU i = top; top = ss[-i]),
    /// @}
    /// @defgroup Stack ops - double
    /// @{
    CODE("2dup", PUSH(ss[-1]); PUSH(ss[-1])),
    CODE("2drop",ss.pop(); top = ss.pop()),
    CODE("2over",PUSH(ss[-3]); PUSH(ss[-3])),
    CODE("2swap",
        DU n = ss.pop(); DU m = ss.pop(); DU l = ss.pop();
        ss.push(n); PUSH(l); PUSH(m)),
    /// @}
    /// @defgroup FPU/ALU ops
    /// @{
    CODE("+",    top += ss.pop()),
    CODE("*",    top *= ss.pop()),
    CODE("-",    top =  ss.pop() - top),
    CODE("/",    top =  ss.pop() / top),
    CODE("mod",  top =  INT(ss.pop()) % INT(top)),
    CODE("*/",   top =  ss.pop() * ss.pop() / top),
    CODE("/mod",
        DU n = ss.pop(); DU t = top;
        ss.push(INT(n) % INT(t)); top = (n / t)),
    CODE("*/mod",
        DU n = ss.pop() * ss.pop();
        DU t = top;
        ss.push(INT(n) % INT(t)); top = (n / t)),
    CODE("and",  top = INT(ss.pop()) & INT(top)),
    CODE("or",   top = INT(ss.pop()) | INT(top)),
    CODE("xor",  top = INT(ss.pop()) ^ INT(top)),
    CODE("abs",  top = abs(top)),
    CODE("negate", top = -top),
    CODE("max",  DU n=ss.pop(); top = (top>n)?top:n),
    CODE("min",  DU n=ss.pop(); top = (top<n)?top:n),
    CODE("2*",   top *= 2),
    CODE("2/",   top /= 2),
    CODE("1+",   top += 1),
    CODE("1-",   top -= 1),
    /// @}
    /// @defgroup Logic ops
    /// @{
    CODE("0= ",  top = BOOL(top == 0)),
    CODE("0<",   top = BOOL(top <  0)),
    CODE("0>",   top = BOOL(top >  0)),
    CODE("=",    top = BOOL(ss.pop() == top)),
    CODE(">",    top = BOOL(ss.pop() >  top)),
    CODE("<",    top = BOOL(ss.pop() <  top)),
    CODE("<>",   top = BOOL(ss.pop() != top)),
    CODE(">=",   top = BOOL(ss.pop() >= top)),
    CODE("<=",   top = BOOL(ss.pop() <= top)),
    /// @}
    /// @defgroup IO ops
    /// @{
    CODE("base@",   PUSH(base)),
    CODE("base!",   fout << setbase(base = POP())),
    CODE("hex",     fout << setbase(base = 16)),
    CODE("decimal", fout << setbase(base = 10)),
    CODE("cr",      fout << ENDL),
    CODE(".",       fout << POP() << " "),
    CODE(".r",      DU n = POP(); dot_r(n, POP())),
    CODE("u.r",     DU n = POP(); dot_r(n, abs(POP()))),
    CODE(".f",      DU n = POP(); fout << setprec(n) << POP()),
    CODE("key",     PUSH(next_word()[0])),
    CODE("emit",    char b = (char)POP(); fout << b),
    CODE("space",   fout << " "),
    CODE("spaces",  for (DU n = POP(), i = 0; i < n; i++) fout << " "),
    /// @}
    /// @defgroup Literal ops
    /// @{
    CODE("[",       compile = false),
    CODE("]",       compile = true),
    IMMD("(",       scan(')')),
    IMMD(".(",      fout << scan(')')),
    CODE("\\",      scan('\n')),
    CODE("$\"",
        const char *s = scan('"')+1;        // string skip first blank
        add_iu(DOSTR);                      // dostr, (+parameter field)
        add_str(s)),                        // byte0, byte1, byte2, ..., byteN
    IMMD(".\"",
        const char *s = scan('"')+1;        // string skip first blank
        add_iu(DOTSTR);                     // dostr, (+parameter field)
        add_str(s)),                        // byte0, byte1, byte2, ..., byteN
    /// @}
    /// @defgroup Branching ops
    /// @brief - if...then, if...else...then
    /// @{
    IMMD("if",      add_iu(ZBRAN); PUSH(XIP); add_iu(0)),        // if    ( -- here )
    IMMD("else",                                                 // else ( here -- there )
        add_iu(BRAN);
        IU h=XIP;   add_iu(0); SETJMP(POP()) = XIP; PUSH(h)),
    IMMD("then",    SETJMP(POP()) = XIP),                        // backfill jump address
    /// @}
    /// @defgroup Loops
    /// @brief  - begin...again, begin...f until, begin...f while...repeat
    /// @{
    IMMD("begin",   PUSH(XIP)),
    IMMD("again",   add_iu(BRAN);  add_iu(POP())),               // again    ( there -- )
    IMMD("until",   add_iu(ZBRAN); add_iu(POP())),               // until    ( there -- )
    IMMD("while",   add_iu(ZBRAN); PUSH(XIP); add_iu(0)),        // while    ( there -- there here )
    IMMD("repeat",  add_iu(BRAN);                                // repeat    ( there1 there2 -- )
        IU t=POP(); add_iu(POP()); SETJMP(t) = XIP),             // set forward and loop back address
    /// @}
    /// @defgrouop For loops
    /// @brief  - for...next, for...aft...then...next
    /// @{
    IMMD("for" ,    add_iu(TOR); PUSH(XIP)),                     // for ( -- here )
    IMMD("next",    add_iu(DONEXT); add_iu(POP())),              // next ( here -- )
    IMMD("aft",                                                  // aft ( here -- here there )
        POP(); add_iu(BRAN);
        IU h=XIP; add_iu(0); PUSH(XIP); PUSH(h)),
    /// @}
    /// @defgrouop Compiler ops
    /// @{
    CODE(":", colon(next_word()); compile=true),
    IMMD(";", compile = false),
    CODE("variable",                                             // create a variable
        colon(next_word());                                      // create a new word on dictionary
        add_iu(DOVAR);                                           // dovar (+parameter field)
        int n = 0; add_du(n)),                                   // data storage (32-bit integer now)
    CODE("constant",                                             // create a constant
        colon(next_word());                                      // create a new word on dictionary
        add_iu(DOLIT);                                           // dovar (+parameter field)
        add_du(POP())),                                          // data storage (32-bit integer now)
    /// @}
    /// @defgroup metacompiler
    /// @brief - dict is directly used, instead of shield by macros
    /// @{
    CODE("exit",  IP = PFA(WP) + PFLEN(WP)),                     // quit current word execution
    CODE("exec",  CALL(POP())),                                  // execute word
    CODE("create",
        colon(next_word());                                      // create a new word on dictionary
        add_iu(DOVAR)),                                          // dovar (+ parameter field)
    CODE("to",              // 3 to x                            // alter the value of a constant
        IU w = find(next_word());                                // to save the extra @ of a variable
        *(DU*)(PFA(w) + sizeof(IU)) = POP()),
    CODE("is",              // ' y is x                          // alias a word
        IU w = find(next_word());                                // can serve as a function pointer
        dict[POP()].pfa = dict[w].pfa),                          // but might leave a dangled block
    CODE("[to]",            // : xx 3 [to] y ;                   // alter constant in compile mode
        IU w = *(IU*)IP; IP += sizeof(IU);                       // fetch constant pfa from 'here'
        *(DU*)(PFA(w) + sizeof(IU)) = POP()),
    ///
    /// be careful with memory access, especially BYTE because
    /// it could make access misaligned which slows the access speed by 2x
    ///
    CODE("@",     IU w = POP(); PUSH(CELL(w))),                  // w -- n
    CODE("!",     IU w = POP(); CELL(w) = POP();),               // n w --
    CODE(",",     DU n = POP(); add_du(n)),
    CODE("allot", DU v = 0; for (IU n = POP(), i = 0; i < n; i++) add_du(v)), // n --
    CODE("+!",    IU w = POP(); CELL(w) += POP()),               // n w --
    CODE("?",     IU w = POP(); fout << CELL(w) << " "),         // w --
    /// @}
    /// @defgroup Debug ops
    /// @{
    CODE("here",  PUSH(HERE)),
    CODE("ucase", ucase = POP()),
    CODE("words", words()),
    CODE("'",     IU w = find(next_word()); PUSH(w)),
    CODE(".s",    ss_dump()),
    CODE("see",   IU w = find(next_word()); IU ip=0; see(&w, &ip)),
    CODE("dump",  DU n = POP(); IU a = POP(); mem_dump(a, INT(n))),
    CODE("forget",
        int w = find(next_word());
        if (w<0) return;
        IU b = find("boot")+1;
        dict.clear(w > b ? w : b)),
#if ARDUINO
    /// @}
    /// @defgroup Arduino specific ops
    /// @{
    CODE("pin",   DU p = POP(); pinMode(p, POP())),
    CODE("in",    PUSH(digitalRead(POP()))),
    CODE("out",   DU p = POP(); digitalWrite(p, POP())),
    CODE("adc",   PUSH(analogRead(POP()))),
    CODE("duty",  DU p = POP(); analogWrite(p, POP(), 255)),
    CODE("attach",DU p  = POP(); ledcAttachPin(p, POP())),
    CODE("setup", DU ch = POP(); DU freq=POP(); ledcSetup(ch, freq, POP())),
    CODE("tone",  DU ch = POP(); ledcWriteTone(ch, POP())),
#endif // ARDUINO
    /// @}
    /// @defgroup System ops
    /// @{
    CODE("peek",  DU a = POP(); PUSH(PEEK(a))),
    CODE("poke",  DU a = POP(); POKE(a, POP())),
    CODE("clock", PUSH(millis())),
    CODE("delay", delay(POP())),             // TODO: change to VM_WAIT
    CODE("bye",   status = VM_STOP),
    CODE("boot",  dict.clear(find("boot") + 1); pmem.clear())
    /// @}
    };
    dict.push((Code*)prim, sizeof(prim)/sizeof(Code));
    status = VM_RUN;
    
    printf("init() this=%p sizeof(Code)=%d\n", this, sizeof(Code));
};
///
/// ForthVM Outer interpreter
///
__GPU__ void
ForthVM::outer() {
    while (fin >> idiom) {
        printf("%d>> %s => ", blockIdx.x, idiom);
        int w = find(idiom);                 /// * search through dictionary
        if (w>=0) {                          /// * word found?
            printf("[%d]:%s %p\n", w, dict[w].name, dict[w].xt);
            if (compile && !dict[w].immd) {  /// * in compile mode?
                add_iu(w);                   /// * add found word to new colon word
            }
            else {
                printf(" call %p", dict[w].xt);
            	CALL(w);
            }                /// * execute forth word
            continue;
        }
        // try as a number
        char *p;
        int n = INT(STRTOL(idiom, &p, base));
        printf("%d\n", n);
        if (*p != '\0') {                    /// * not number
            fout << idiom << "? " << ENDL;   ///> display error prompt
            compile = false;                 ///> reset to interpreter mode
            break;                           ///> skip the entire input buffer
        }
        // is a number
        if (compile) {                       /// * add literal when in compile mode
            add_iu(DOLIT);                   ///> dovar (+parameter field)
            add_du(n);                       ///> data storage (32-bit integer now)
        }
        else PUSH(n);                        ///> or, add value onto data stack
    }
    if (!compile) ss_dump();
}
//=======================================================================================
