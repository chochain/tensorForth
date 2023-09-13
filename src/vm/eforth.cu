/** -*- c++ -*-
 * @file
 * @brief ForthVM class - eForth VM implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "eforth.h"
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
///@name Dictioanry access
///@{
#define PFA(w)    (dict[(IU)(w)].pfa)           /**< PFA of given word id                    */
#define HERE      (mmu.here())                  /**< current context                         */
#define SETJMP(a) (mmu.setjmp(a))               /**< address offset for branching opcodes    */
///@}
///@name Heap memory load/store macros
///@{
#define LDi(a)    (mmu.ri((IU)(a)))             /**< read an instruction unit from pmem      */
#define LDd(a)    (mmu.rd((IU)(a)))             /**< read a data unit from pmem              */
#define LDp(a)    (mmu.pmem((IU)(a)))
#define STi(a, d) (mmu.wi((IU)(a), (IU)(d)))    /**< write a instruction unit to pmem        */
#define STd(a,d) (mmu.wd((IU)(a), (DU)(d)))     /**< write a data unit to pmem               */
///@}
///
/// resume suspended task
///
__GPU__ int
ForthVM::resume() {
    VLOG1("VM[%d] resumed at WP=%d, IP=%d\n", vid, WP, IP);
    nest();           /// * will set state to VM_READY
    return 1;         /// * OK, continue to outer loop
}

__GPU__ int
ForthVM::post() {
    cudaError_t code = cudaGetLastError();
    
    if (code == cudaSuccess) return 0;
    
    ERROR("VM ERROR: %s %s %d, IP=%d, w=%d\n",
          cudaGetErrorString(code), __FILE__, __LINE__, IP, LDi(IP));
    ss_dump();
    state = VM_WAIT;
    return 1;
}
///
/// Forth inner interpreter (colon word handler)
/// Note: nest is also used for resume where RS != 0
///
#if CC_DEBUG
#define _log(hdr)  printf("%03d: %s.rs=%d\n", IP, hdr, rs.idx)
#define _dlog(w)   printf("\t%03d: %s %d\n", IP, dict[w].name, w)
#else
#define _log(hdr)
#define _dlog(w)
#endif // CC_DEBUG
__GPU__ void
ForthVM::nest() {
    ///
    /// when IP != 0, it resumes paused VM
    ///
    _log("IN");
    while (state == VM_RUN && IP) {                  /// * try no recursion
        IU w = LDi(IP);                              ///< fetch opcode, and cache dataline hopefully
        IP += sizeof(IU);
        _dlog(w);
        if (dict[w].colon) {                         ///< is it a colon word?
            rs.push(WP);                             /// * ENTER
            rs.push(IP);                             /// * setup call frame 
            IP = dict[w].pfa;                        ///< jump to pfa of given colon word
            _log("ENTER");
        }
        else if (w==EXIT) {                          /// * EXIT
            IP = INT(rs.pop());                      /// * restore call frame
            WP = INT(rs.pop());
            yield();                                 /// * for multi-tasking
            _log("EXIT");
        }
        else if (w==DONEXT && !IS_OBJ(rs[-1])) {     ///< DONEXT handler
            /// save 600ms / 100M cycles on Intel
            if ((rs[-1] -= 1) >= -DU_EPS) IP = LDi(IP); ///< decrement loop counter, and fetch target addr
            else { IP += sizeof(IU); rs.pop(); }        ///< done loop, pop off loop counter
            _log("DONEXT");
        }
        else (*dict[w].xt)();                        ///< execute primitive word
    }
    _log("OUT");
    if (state == VM_RUN) state = VM_READY;           /// * READY for next input
}
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    Code &c = dict[w];                               /// * code reference
    if (c.colon) {                                   /// * userd defined word
//        printf("%03d WP=%d CALL[%d] %s\n", IP, WP, w, c.name);
        rs.push(WP);                                 /// * setup call frame
        rs.push(IP=0);
        WP    = w;                                   /// * frame for new word
        IP    = c.pfa;
        state = VM_RUN;
        nest();                                      /// * Forth inner loop
    }
    else {
        UFP xt = (UFP)c.xt & ~CODE_ATTR_FLAG;        /// * strip off immediate bit
        (*(FPTR)xt)();                               /// * execute function
    }
}
///
/// Dictionary compiler proxy macros to reduce verbosity
///
__GPU__ void ForthVM::add_w(IU w)  {
    add_iu(w);
    VLOG2("add_w(%d) => %s\n", w, dict[w].name);
}
__GPU__ void ForthVM::add_iu(IU i) { mmu.add((U8*)&i, sizeof(IU)); }
__GPU__ void ForthVM::add_du(DU d) { mmu.add((U8*)&d, sizeof(DU)); }
__GPU__ void ForthVM::add_str(const char *s, bool adv) {
    int sz = STRLENB(s)+1; sz = ALIGN2(sz);           ///> calculate string length, then adjust alignment (combine?)
    mmu.add((U8*)s, sz, adv);
}
///
/// dictionary initializer
///
__GPU__ void
ForthVM::init() {
    const Code prim[] = {       /// singleton, build once only
    ///
    ///@defgroup Execution flow ops
    ///@brief - DO NOT change the sequence here (see forth_opcode enum)
    ///@{
    CODE("exit",    {}),                                  /// * quit word, handled in nest()
    CODE("donext",  {}),                                  /// * handled in nest(),
//         if ((rs[-1] -= 1) >= -DU_EPS) IP = LDi(IP);    /// * also overwritten in netvm later
//         else { IP += sizeof(IU); rs.pop(); }),
    CODE("dovar",   PUSH(IP); IP += sizeof(DU)),
    CODE("dolit",   PUSH(LDd(IP)); IP += sizeof(DU)),
    CODE("dostr",
        char *s  = (char*)LDp(IP);                        // get string ptr & len
        int   sz = STRLENB(s)+1;                          // '\0' terminated
        PUSH(IP); PUSH(sz-1); IP += ALIGN2(sz)),
    CODE("dotstr",
        char *s  = (char*)LDp(IP);                        // get string pointer
        int  sz  = STRLENB(s)+1;
        fout << s;  IP += ALIGN2(sz)),                    // send to output console
    CODE("branch" , IP = LDi(IP)),                        // unconditional branch
    CODE("0branch", IP = ZERO(POP()) ? LDi(IP) : IP + sizeof(IU)), // conditional branch
    CODE("does",                                          // CREATE...DOES... meta-program
         IU ip = PFA(WP);
         while (LDi(ip) != DOES) ip++;                    // find DOES
         while (LDi(ip)) add_iu(LDi(ip))),                // copy&paste code
    CODE(">r",   rs.push(POP())),
    CODE("r>",   PUSH(rs.pop())),
    CODE("r@",   PUSH(mmu.dup(rs[-1]))),
    ///@}
    ///@defgroup Stack ops
    ///@brief - opcode sequence can be changed below this line
    ///@{
    CODE("dup",  PUSH(mmu.dup(top))),                     // CC: new view created
    CODE("drop", mmu.drop(top); top = ss.pop()),          // free tensor or view
    CODE("over", PUSH(mmu.dup(ss[-1]))),                  // CC: new view created
    CODE("swap", DU n = ss.pop(); PUSH(n)),
    CODE("rot",  DU n = ss.pop(); DU m = ss.pop(); ss.push(n); PUSH(m)),
    CODE("pick", int i = INT(top); top = mmu.dup(ss[-i])),
    ///@}
    ///@defgroup Stack double
    ///@{
    CODE("2dup", PUSH(mmu.dup(ss[-1])); PUSH(mmu.dup(ss[-1]))),
    CODE("2drop",
         DU s = ss.pop(); mmu.drop(s); mmu.drop(top);
         top = ss.pop()),
    CODE("2over",PUSH(mmu.dup(ss[-3])); PUSH(mmu.dup(ss[-3]))),
    CODE("2swap",
        DU n = ss.pop(); DU m = ss.pop(); DU l = ss.pop();
        ss.push(n); PUSH(l); PUSH(m)),
    ///@}
    ///@defgroup FPU ops
    ///@{
    CODE("+",    top = ADD(top, ss.pop()); SCALAR(top)),
    CODE("*",    top = MUL(top, ss.pop()); SCALAR(top)),
    CODE("-",    top = SUB(ss.pop(), top); SCALAR(top)),
    CODE("/",    top = DIV(ss.pop(), top); SCALAR(top)),
    CODE("mod",  top = MOD(ss.pop(), top); SCALAR(top)),  /// fmod = x - int(q)*y
    CODE("/mod",
        DU n = ss.pop();
        DU m = MOD(n, top); ss.push(SCALAR(m));
        top = DIV(n, top); SCALAR(top)),
    ///@}
    ///@defgroup FPU double precision ops
    ///@{
    CODE("*/",   top = (DU2)ss.pop() * ss.pop() / top; SCALAR(top)),
    CODE("*/mod",
        DU2 n = (DU2)ss.pop() * ss.pop();
        DU  m = MOD(n, top); ss.push(SCALAR(m));
        top = round(n / top)),
    ///@}
    ///@defgroup Binary logic ops (convert to integer first)
    ///@{
    CODE("and",  top = I2D(INT(ss.pop()) & INT(top))),
    CODE("or",   top = I2D(INT(ss.pop()) | INT(top))),
    CODE("xor",  top = I2D(INT(ss.pop()) ^ INT(top))),
    CODE("abs",  top = ABS(top)),
    CODE("negate", top = MUL(top, -DU1)),
    CODE("max",  DU n=ss.pop(); top = MAX(top, n)),
    CODE("min",  DU n=ss.pop(); top = MIN(top, n)),
    ///@}
    ///@defgroup Data conversion ops
    ///@{
    CODE("int",  top = INT(top)),                /// integer part, 1.5 => 1, -1.5 => -1
    CODE("round",top = round(top)),              /// rounding 1.5 => 2, -1.5 => -1
    CODE("ceil", top = ceil(top)),
    CODE("floor",top = floor(top)),
    ///@}
    ///@defgroup Logic ops
    ///@{
    CODE("0= ",  top = BOOL(ZERO(top))),
    CODE("0<",   top = BOOL(top < -DU_EPS)),
    CODE("0>",   top = BOOL(top > DU_EPS)),
    CODE("=",    top = BOOL(ZERO(ss.pop() - top))),
    CODE("<",    top = BOOL((ss.pop() - top) < -DU_EPS)),
    CODE(">",    top = BOOL((ss.pop() - top) > DU_EPS)),
    CODE("<>",   top = BOOL(!ZERO(ss.pop() - top))),
    CODE("<=",   top = BOOL(INT(ss.pop()) <= INT(top))),    /// int, for count or loop control
    CODE(">=",   top = BOOL(INT(ss.pop()) >= INT(top))),
    ///@}
    ///@defgroup IO ops
    ///@{
    CODE("base@",   PUSH(I2D(radix))),
    CODE("base!",   fout << setbase(radix = POPi)),
    CODE("hex",     fout << setbase(radix = 16)),
    CODE("decimal", fout << setbase(radix = 10)),
    CODE("cr",      fout << ENDL),
    CODE(".",       dot(POP())),
    CODE(".r",      int n = POPi; dot_r(n, POP())),
    CODE("u.r",     int n = POPi; dot_r(n, ABS(POP()))),
    CODE(".f",      int n = POPi; fout << setprec(n) << POP()),
    CODE("key",     PUSH(next_idiom()[0])),
    CODE("emit",    fout << (char)POPi),
    CODE("space",   fout << ' '),
    CODE("spaces",
         int n = POPi;
         MEMSET(idiom, ' ', n); idiom[n] = '\0';
         fout << idiom),
    CODE("type",
         int n = POPi; int idx = POPi;
         fout << (char*)LDp(idx)),          // get string pointer
    ///@}
    ///@defgroup Literal ops
    ///@{
    CODE("[",       compile = false),
    CODE("]",       compile = true),
    IMMD("(",       scan(')')),
    IMMD(".(",      fout << scan(')')),
    IMMD("\\",      scan('\n')),
    IMMD("s\"",
        const char *s = scan('"')+1;        // string skip first blank
        if (compile) add_w(DOSTR);          // dostr, (+parameter field)
        else {
            PUSH(HERE);
            PUSH(STRLENB(s)+1);
        }
        add_str(s, compile)),               // string on PAD in interpreter mode
    IMMD(".\"",
        const char *s = scan('"')+1;        // string skip first blank
        if (compile) {
            add_w(DOTSTR);                  // dotstr, (+parameter field)
            add_str(s);
        }
        else fout << s),                    // print right away
    ///@}
    ///@defgroup Branching ops
    ///@brief - if...then, if...else...then
    ///@{
    IMMD("if", add_w(ZBRAN); PUSH(HERE); add_iu(0)),        // if   ( -- here )
    IMMD("else",                                            // else ( here -- there )
        add_w(BRAN);
        IU h = HERE; add_iu(0); SETJMP(POPi); PUSH(h)),     // set forward jump
    IMMD("then", SETJMP(POPi)),                             // backfill jump address
    ///@}
    ///@defgroup Loop ops
    ///@brief  - begin...again, begin...f until, begin...f while...repeat
    ///@{
    IMMD("begin",   PUSH(HERE)),
    IMMD("again",   add_w(BRAN);  add_iu(POPi)),            // again    ( there -- )
    IMMD("until",   add_w(ZBRAN); add_iu(POPi)),            // until    ( there -- )
    IMMD("while",   add_w(ZBRAN); PUSH(HERE); add_iu(0)),   // while    ( there -- there here )
    IMMD("repeat",  add_w(BRAN);                            // repeat    ( there1 there2 -- )
        IU t=POPi; add_iu(POPi); SETJMP(t)),                // set forward and loop back address
    ///@}
    ///@defgrouop For-loop ops
    ///@brief  - for...next, for...aft...then...next
    ///@{
    IMMD("for" ,    add_w(TOR); PUSH(HERE)),                // for ( -- here )
    IMMD("next",    add_w(DONEXT); add_iu(POPi)),           // next ( here -- )
    IMMD("aft",                                             // aft ( here -- here there )
        POP(); add_w(BRAN);
        IU h=HERE; add_iu(0); PUSH(HERE); PUSH(h)),
    ///@}
    ///@defgrouop Compiler ops
    ///@{
    CODE(":", mmu.colon(next_idiom()); compile=true),
    IMMD(";", add_w(EXIT); compile = false),                // terminate a word
    CODE("variable",                                        // create a variable
        mmu.colon(next_idiom());                            // create a new word on dictionary
        add_w(DOVAR);                                       // dovar (+parameter field)
        add_du(0);                                          // data storage (32-bit float now)
        add_w(EXIT)),
    CODE("constant",                                        // create a constant
        mmu.colon(next_idiom());                            // create a new word on dictionary
        add_w(DOLIT);                                       // dovar (+parameter field)
        add_du(POP());
        add_w(EXIT)),
    ///@}
    ///@defgroup word defining words (DSL)
    ///@brief - dict is directly used, instead of shield by macros
    ///@{
    CODE("exec",  call(POPi)),                              // execute word
    CODE("create",
        mmu.colon(next_idiom());                            // create a new word on dictionary
        add_w(DOVAR)),                                      // dovar (+ parameter field)
    CODE("to",              // 3 to x                       // alter the value of a constant
        int w = FIND(next_idiom());                         // to save the extra @ of a variable
        IU  a = PFA(w) + sizeof(IU);
        DU  d = POP();
        if (a < T4_PMEM_SZ) STd(a, d);                      // store TOS to constant's pfa
        else ERROR("to %x", a)),
    CODE("is",              // ' y is x                     // alias a word
        int w = FIND(next_idiom());                         // can serve as a function pointer
        IU  a = PFA(POPi);
        IU  i = PFA(w);
        if (a < T4_PMEM_SZ) STi(a, i);                     // point x to y
        else { ERROR("is %x", a); state = VM_STOP; }),
    CODE("[to]",            // : xx 3 [to] y ;              // alter constant in compile mode
        IU w = LDi(IP); IP += sizeof(IU);                   // fetch constant pfa from 'here'
        IU a = PFA(w) + sizeof(IU);
        DU d = POP();
        if (a < T4_PMEM_SZ) STd(a, d);                      // store TOS into constant pfa
        else { ERROR("is %x", a); state = VM_STOP; }),
    ///
    /// be careful with memory access, because
    /// it could make access misaligned which cause exception
    ///
    CODE("C@",    IU w = POPi; PUSH(*(char*)LDp(w))),
    CODE("C!",    IU w = POPi; DU n = POP(); *((char*)LDp(w)) = (U8)n),
    CODE("@",     IU w = POPi; PUSH(LDd(w))),                                     // w -- n
    CODE("!",     IU w = POPi; STd(w, POP())),                                    // n w --
    CODE(",",     DU n = POP(); add_du(n)),
    CODE("allot", DU v = 0; for (IU n = POPi, i = 0; i < n; i++) add_du(v)),      // n --
    CODE("+!",    IU w = POPi; DU v = ADD(LDd(w), POP()); STd(w, SCALAR(v))),     // n w --
    CODE("?",     IU w = POPi; fout << LDd(w) << " "),                            // w --
    ///@}
    ///@defgroup Debug ops
    ///@{
    CODE("here",  PUSH(HERE)),
    CODE("ucase", ucase = !ZERO(POPi)),
    CODE("'",     int w = FIND(next_idiom()); PUSH(w)),
    CODE("didx",  IU w = POPi; PUSH(dict[w].didx)),
    CODE("pfa",   IU w = POPi; PUSH(PFA(w))),
    CODE("nfa",   IU w = POPi; PUSH(dict[w].nfa)),
    CODE("trace", mmu.trace(POPi)),                                               // turn tracing on/off
    CODE(".s",    ss_dump()),
    CODE("words", fout << opx(OP_WORDS)),
    CODE("see",   int w = FIND(next_idiom()); fout << opx(OP_SEE, w)),
    CODE("dump",  DU n = POP(); int a = POPi; fout << opx(OP_DUMP, a, n)),
    CODE("forget",
        int w = FIND(next_idiom());
        if (w<0) return;
        IU b = FIND("boot")+1;
        mmu.clear(w > b ? w : b)),
    ///@}
    ///@defgroup System ops
    ///@{
    CODE("mstat",
         int t = mmu.trace();
         mmu.trace(1);
         mmu.status();
         mmu.trace(t)),
    CODE("clock", DU t = mmu.ms(); SCALAR(t); PUSH(t)),
    CODE("delay", delay(POPi)),                  ///< TODO: change to VM_WAIT
    CODE("pause", state = VM_WAIT),              ///< yield to other VM
    CODE("bye",   state = VM_STOP),
    ///@}
    CODE("boot",  mmu.clear(FIND("boot") + 1))
    };
    VM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); ///< append dictionary
    VLOG1("ForthVM::init ok\n");
};
///
/// ForthVM Outer interpreter
/// @brief having outer() on device creates branch divergence but
///    + can enable parallel VMs (with different tasks)
///    + can support parallel find()
///    + can support system without a host
///    However, to optimize,
///    + compilation can be done on host and
///    + only call() is dispatched to device
///    + number() and find() can run in parallel
///    - however, find() can run in serial only
///
/// parse input idiom as a word
///
__GPU__ int
ForthVM::parse(char *str) {
    int w = FIND(str);                    /// * search through dictionary
    if (w < 0) {                          /// * input word not found
        VLOG2("'%s' not found\n", str);
        return 0;                         /// * next, try as a number
    }
    VLOG2("%4x:%p %s %d ",
        dict[w].colon ? dict[w].pfa : 0, dict[w].xt, dict[w].name, w);
    if (compile && !dict[w].immd) {       /// * in compile mode?
        add_w((IU)w);                     /// * add found word to new colon word
    }
    else {
        VLOG2("=> call(%s)\n", dict[w].name);
        call((IU)w);                      /// * execute forth word
    }
    return 1;
}
///
/// parse input idiom as a number
///
__GPU__ int
ForthVM::number(char *str) {
    char *p;
    DU n = (STRCHR(idiom, '.'))
        ? STRTOF(idiom, &p)
        : STRTOL(idiom, &p, radix);
    if (*p != '\0') return 0;            /// * not a number, bail
    // is a number
    if (compile) {                       /// * add literal when in compile mode
        VLOG2("%d| %f\n", vid, n);
        add_w(DOLIT);                    ///> dovar (+parameter field)
        add_du(n);                       ///> store literal
    }
    else {                               ///> or, add value onto data stack
        VLOG2("%d| ss.push(%f)\n", vid, n);
        PUSH(n);
    }
    return 1;
}
//=======================================================================================
