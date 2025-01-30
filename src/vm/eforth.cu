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
#define MU        (sys->mu)
#define PFA(w)    (dict[(IU)(w)].pfa)           /**< PFA of given word id                    */
#define HERE      (MU->here())                  /**< current context                         */
#define SETJMP(a) (MU->setjmp(a))               /**< address offset for branching opcodes    */
///@}
///@name Heap memory load/store macros
///@{
#define LDi(a)    (MU->ri((IU)(a)))              /**< read an instruction unit from pmem      */
#define LDd(a)    (MU->rd((IU)(a)))              /**< read a data unit from pmem              */
#define LDp(a)    (MU->pmem((IU)(a)))
#define STi(a,d)  (MU->wi((IU)(a), (IU)(d)))     /**< write a instruction unit to pmem        */
#define STd(a,d)  (MU->wd((IU)(a), (DU)(d)))     /**< write a data unit to pmem               */
#define STc(a,c)  (*((char*)LDp(w))=(U8)INT(c))  /**< write a char to pmem                    */

__GPU__
ForthVM::ForthVM(int id, System *sys) : VM(id, sys) {
    dict = sys->mu->dict();
    base = sys->mu->pmem(id);
//        VLOG1("\\  ::ForthVM[%d](dict=%p) sizeof(Code)=%ld\n", id, dict, sizeof(Code));
    printf("\\ ::ForthVM[%d](dict=%p) sizeof(Code)=%ld\n", id, dict, sizeof(Code));
}
///
/// resume suspended task
///
__GPU__ int
ForthVM::resume() {
    VLOG1("VM[%d] resumed at WP=%d, IP=%d\n", id, WP, IP);
    nest();           /// * will set state to VM_READY
    return 1;         /// * OK, continue to outer loop
}
/* Note: for debugging, add post() to eforth.h
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
*/
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
    while (state == NEST && IP) {                    /// * try no recursion
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
            if (GT(rs[-1] -= 1, -DU1)) IP = LDi(IP); ///< decrement loop counter, and fetch target addr
            else { IP += sizeof(IU); rs.pop(); }     ///< done loop, pop off loop counter
            _log("DONEXT");
        }
        else (*dict[w].xt)();                        ///< execute primitive word
    }
    _log("OUT");
    if (state == NEST) state = HOLD;                 /// * READY for next input
}
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    Code &c = dict[w];                               /// * code reference
    if (c.colon) {                                   /// * userd defined word
//        printf("%03d WP=%d CALL[%d] %s\n", IP, WP, w, c.name);
        rs.push(WP);                                 /// * setup call frame
        rs.push(IP=0);
        WP    = w;                                   /// * frame for new word
        IP    = c.pfa;
        state = NEST;
        nest();                                      /// * Forth inner loop
    }
    else {
        UFP xt = (UFP)c.xt & ~CODE_ATTR_FLAG;        /// * strip off attribute bit
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
__GPU__ void ForthVM::add_iu(IU i) { MU->add((U8*)&i, sizeof(IU)); }
__GPU__ void ForthVM::add_du(DU d) { MU->add((U8*)&d, sizeof(DU)); }
__GPU__ void ForthVM::add_str(const char *s, bool adv) {
    int sz = STRLENB(s)+1; sz = ALIGN2(sz);           ///> calculate string length, then adjust alignment (combine?)
    MU->add((U8*)s, sz, adv);
}
///
/// dictionary initializer
///
__GPU__ void
ForthVM::init() {
    VM::init();
    printf("\\ ::ForthVM[%d].init() ok\n", id);
    return;
    
    dict = sys->mu->dict();
    base = sys->mu->pmem(id);
    ///
    ///@defgroup Execution flow ops
    ///@brief - DO NOT change the sequence here (see forth_opcode enum)
    ///@{
    CODE("exit",    {});       /// * quit word, handled in nest()
    CODE("donext",  {});                                  /// * handled in nest(),
    // if (GT(rs[-1] -= 1, -DU1)) IP = LDi(IP);           /// * also overwritten in netvm later
    // else { IP += sizeof(IU); rs.pop(); });
    CODE("dovar",   PUSH(IP); IP += sizeof(DU));
    CODE("dolit",   PUSH(LDd(IP)); IP += sizeof(DU));
    CODE("dostr",
        char *s  = (char*)LDp(IP);                        // get string ptr & len
        int   sz = STRLENB(s)+1;                          // '\0' terminated
        PUSH(IP); PUSH(sz-1); IP += ALIGN2(sz));
    CODE("dotstr",
        char *s  = (char*)LDp(IP);                        // get string pointer
        int  sz  = STRLENB(s)+1;
        sys->pstr(s);  IP += ALIGN2(sz));                 // send to output console
    CODE("branch" , IP = LDi(IP));                        // unconditional branch
    CODE("0branch", IP = ZEQ(POP()) ? LDi(IP) : IP + sizeof(IU)); // conditional branch
    CODE("does",                                          // CREATE...DOES... meta-program
         IU ip = PFA(WP);
         while (LDi(ip) != DOES) ip++;                    // find DOES
         while (LDi(ip)) add_iu(LDi(ip)));                // copy&paste code
    CODE(">r",   rs.push(POP()));
    CODE("r>",   PUSH(rs.pop()));
    CODE("r@",   PUSH(MU->dup(rs[-1])));
    ///@}
    ///@defgroup Stack ops
    ///@brief - opcode sequence can be changed below this line
    ///@{
    CODE("dup",  PUSH(MU->dup(tos)));                     // CC: new view created
    CODE("drop", MU->drop(tos); tos = ss.pop());          // free tensor or view
    CODE("over", PUSH(MU->dup(ss[-1])));                  // CC: new view created
    CODE("swap", DU n = ss.pop(); PUSH(n));
    CODE("rot",  DU n = ss.pop(); DU m = ss.pop(); ss.push(n); PUSH(m));
    CODE("pick", int i = INT(tos); tos = MU->dup(ss[-i]));
    ///@}
    ///@defgroup Stack double
    ///@{
    CODE("2dup", PUSH(MU->dup(ss[-1])); PUSH(MU->dup(ss[-1])));
    CODE("2drop",
         DU s = ss.pop(); MU->drop(s); MU->drop(tos);
         tos = ss.pop());
    CODE("2over",PUSH(MU->dup(ss[-3])); PUSH(MU->dup(ss[-3])));
    CODE("2swap",
        DU n = ss.pop(); DU m = ss.pop(); DU l = ss.pop();
         ss.push(n); PUSH(l); PUSH(m));
    ///@}
    ///@defgroup FPU ops
    ///@{
    CODE("+",    tos = ADD(tos, ss.pop()); SCALAR(tos));
    CODE("*",    tos = MUL(tos, ss.pop()); SCALAR(tos));
    CODE("-",    tos = SUB(ss.pop(), tos); SCALAR(tos));
    CODE("/",    tos = DIV(ss.pop(), tos); SCALAR(tos));
    CODE("mod",  tos = MOD(ss.pop(), tos); SCALAR(tos));  /// fmod = x - int(q)*y
    CODE("/mod",
        DU n = ss.pop();
        DU m = MOD(n, tos); ss.push(SCALAR(m));
        tos = DIV(n, tos); SCALAR(tos));
    ///@}
    ///@defgroup FPU double precision ops
    ///@{
    CODE("*/",   tos = (DU2)ss.pop() * ss.pop() / tos; SCALAR(tos));
    CODE("*/mod",
        DU2 n = (DU2)ss.pop() * ss.pop();
        DU  m = MOD(n, tos); ss.push(SCALAR(m));
        tos = round(n / tos));
    ///@}
    ///@defgroup Binary logic ops (convert to integer first)
    ///@{
    CODE("and",  tos = I2D(INT(ss.pop()) & INT(tos)));
    CODE("or",   tos = I2D(INT(ss.pop()) | INT(tos)));
    CODE("xor",  tos = I2D(INT(ss.pop()) ^ INT(tos)));
    CODE("abs",  tos = ABS(tos));
    CODE("negate", tos = MUL(tos, -DU1));
    CODE("max",  DU n=ss.pop(); tos = MAX(tos, n));
    CODE("min",  DU n=ss.pop(); tos = MIN(tos, n));
    ///@}
    ///@defgroup Data conversion ops
    ///@{
    CODE("int",  tos = INT(tos));    /// nearest-even 0.5 => 0, 1.5 => 2, 2.5 => 2
    CODE("round",tos = round(tos));  /// 0.5 => 1, 1.5 => 2, 2.5 => 3, 1.5 => -2 
    CODE("ceil", tos = ceil(tos));   /// 1.5 => 2, -1.5 => -1
    CODE("floor",tos = floor(tos));  /// 1.5 => 1, -1.5 => -2
    ///@}
    ///@defgroup Logic ops
    ///@{
    CODE("0= ",  tos = BOOL(ZEQ(tos)));
    CODE("0<",   tos = BOOL(LT(tos, DU0)));
    CODE("0>",   tos = BOOL(GT(tos, DU0)));
    CODE("=",    tos = BOOL(EQ(ss.pop(), tos)));
    CODE("<",    tos = BOOL(LT(ss.pop(), tos)));
    CODE(">",    tos = BOOL(GT(ss.pop(), tos)));
    CODE("<>",   tos = BOOL(!EQ(ss.pop(), tos)));
    CODE("<=",   tos = BOOL(!GT(ss.pop(), tos)));
    CODE(">=",   tos = BOOL(!LT(ss.pop(), tos)));
    ///@}
    ///@defgroup IO ops
    ///@{
    CODE("base",    PUSH((DU)(base - MU->pmem(0))));
    CODE("decimal", sys->dot(RDX, *base=10));
    CODE("hex",     sys->dot(RDX, *base=16));
    CODE("cr",      sys->dot(CR));
    CODE(".",       sys->dot(DOT,  POP()));
    CODE("u.",      sys->dot(UDOT, POP()));
    CODE(".r",      int n = POPi; sys->dotr(n, POP(), *base));
    CODE("u.r",     int n = POPi; sys->dotr(n, ABS(POP()), *base));
    CODE("key",     PUSH(sys->next_idiom()[0]));
    CODE("emit",    sys->dot(EMIT, POP()));
    CODE("space",   sys->dot(SPCS, DU1));
    CODE("spaces",  sys->dot(SPCS, POP()));
    CODE("type",
         int n = POPi; int idx = POPi;
         sys->pstr((char*)LDp(idx)));             // get string pointer
    ///@}
    ///@defgroup Literal ops
    ///@{
    CODE("[",       compile = false);
    CODE("]",       compile = true);
    IMMD("(",       sys->scan(')'));
    IMMD(".(",      sys->pstr(sys->scan(')')));
    IMMD("\\",      sys->scan('\n'));
    IMMD("s\"",
        const char *s = sys->scan('"')+1;   // string skip first blank
        if (compile) add_w(DOSTR);          // dostr, (+parameter field)
        else {
            PUSH(HERE);
            PUSH(STRLENB(s)+1);
        }
        add_str(s, compile));               // string on PAD in interpreter mode
    IMMD(".\"",
        const char *s = sys->scan('"')+1;   // string skip first blank
        if (compile) {
            add_w(DOTSTR);                  // dotstr, (+parameter field)
            add_str(s);
        }
        else sys->pstr(s));                 // print right away
    ///@}
    ///@defgroup Branching ops
    ///@brief - if...then, if...else...then
    ///@{
    IMMD("if", add_w(ZBRAN); PUSH(HERE); add_iu(0));        // if   ( -- here )
    IMMD("else",                                            // else ( here -- there )
        add_w(BRAN);
        IU h = HERE; add_iu(0); SETJMP(POPi); PUSH(h));     // set forward jump
    IMMD("then", SETJMP(POPi));                             // backfill jump address
    ///@}
    ///@defgroup Loop ops
    ///@brief  - begin...again, begin...f until, begin...f while...repeat
    ///@{
    IMMD("begin",   PUSH(HERE));
    IMMD("again",   add_w(BRAN);  add_iu(POPi));            // again    ( there -- )
    IMMD("until",   add_w(ZBRAN); add_iu(POPi));            // until    ( there -- )
    IMMD("while",   add_w(ZBRAN); PUSH(HERE); add_iu(0));   // while    ( there -- there here )
    IMMD("repeat",  add_w(BRAN);                            // repeat    ( there1 there2 -- )
        IU t=POPi; add_iu(POPi); SETJMP(t));                // set forward and loop back address
    ///@}
    ///@defgrouop For-loop ops
    ///@brief  - for...next, for...aft...then...next
    ///@{
    IMMD("for" ,    add_w(TOR); PUSH(HERE));                // for ( -- here )
    IMMD("next",    add_w(DONEXT); add_iu(POPi));           // next ( here -- )
    IMMD("aft",                                             // aft ( here -- here there )
        POP(); add_w(BRAN);
        IU h=HERE; add_iu(0); PUSH(HERE); PUSH(h));
    ///@}
    ///@defgrouop Compiler ops
    ///@{
    CODE(":", MU->colon(sys->next_idiom()); compile=true);
    IMMD(";", add_w(EXIT); compile = false);                // terminate a word
    CODE("variable",                                        // create a variable
        MU->colon(sys->next_idiom());                       // create a new word on dictionary
        add_w(DOVAR);                                       // dovar (+parameter field)
        add_du(DU0);                                        // data storage (32-bit float now)
        add_w(EXIT));
    CODE("constant",                                        // create a constant
        MU->colon(sys->next_idiom());                       // create a new word on dictionary
        add_w(DOLIT);                                       // dovar (+parameter field)
        add_du(POP());
        add_w(EXIT));
    ///@}
    ///@defgroup word defining words (DSL)
    ///@brief - dict is directly used, instead of shield by macros
    ///@{
    CODE("exec",  call(POPi));                              // execute word
    CODE("create",
        MU->colon(sys->next_idiom());                       // create a new word on dictionary
        add_w(DOVAR));                                      // dovar (+ parameter field)
    CODE("to",              // 3 to x                       // alter the value of a constant
        int w = FIND(sys->next_idiom());                    // to save the extra @ of a variable
        if (w < 0) { ERROR(" word not found"); return; }
        IU  a = PFA(w) + sizeof(IU);
        DU  d = POP();
        if (a < T4_PMEM_SZ) STd(a, d);                      // store TOS to constant's pfa
        else ERROR("to %x", a));
    CODE("is",              // ' y is x                     // alias a word
        int w = FIND(sys->next_idiom());                    // can serve as a function pointer
        if (w < 0) { ERROR(" word not found"); return; }
        IU  a = PFA(POPi);
        IU  i = PFA(w);
        if (a < T4_PMEM_SZ) STi(a, i);                      // point x to y
        else { ERROR("is %x", a); state = STOP; });
    CODE("[to]",            // : xx 3 [to] y ;              // alter constant in compile mode
        IU w = LDi(IP); IP += sizeof(IU);                   // fetch constant pfa from 'here'
        IU a = PFA(w) + sizeof(IU);
        DU d = POP();
        if (a < T4_PMEM_SZ) STd(a, d);                      // store TOS into constant pfa
        else { ERROR("is %x", a); state = STOP; });
    ///
    /// be careful with memory access, because
    /// it could make access misaligned which cause exception
    ///
    CODE("C@",    IU w = POPi; PUSH(*(char*)LDp(w)));
    CODE("C!",    IU w = POPi; DU n = POP(); STc(w, n));
    CODE("@",     IU w = POPi; PUSH(LDd(w)));                                     // w -- n
    CODE("!",     IU w = POPi; STd(w, POP()));                                    // n w --
    CODE(",",     DU n = POP(); add_du(n));
    CODE("allot", DU v = 0; for (IU n = POPi, i = 0; i < n; i++) add_du(v));      // n --
    CODE("+!",    IU w = POPi; DU v = ADD(LDd(w), POP()); STd(w, SCALAR(v)));     // n w --
    CODE("?",     IU w = POPi; sys->dot(DOT, LDd(w)));                            // w --
    ///@}
    ///@defgroup Debug ops
    ///@{
    CODE("here",  PUSH(HERE));
    CODE("'",     int w = FIND(sys->next_idiom()); PUSH(w));
    CODE("didx",  IU w = POPi; PUSH(dict[w].didx));
    CODE("pfa",   IU w = POPi; PUSH(PFA(w)));
    CODE("nfa",   IU w = POPi; PUSH(dict[w].nfa));
    CODE("trace", sys->trace(POPi));                                              // turn tracing on/off
    CODE(".s",    sys->op(OP_SS, id));
    CODE("words", sys->op(OP_WORDS));
    CODE("see",   int w = FIND(sys->next_idiom()); sys->op(OP_SEE, w));
    CODE("dump",  DU n = POP(); int a = POPi; sys->op(OP_DUMP, a, n));
    CODE("forget",
        int w = FIND(sys->next_idiom());
        if (w < 0) return;
        IU b = FIND("boot")+1;
        MU->clear(w > b ? w : b));
    ///@}
    ///@defgroup System ops
    ///@{
    CODE("clock", DU t = sys->ms(); SCALAR(t); PUSH(t));
    CODE("ms",    delay(POPi));                  ///< TODO: change to VM_WAIT
    CODE("pause", state = HOLD);                 ///< yield to other VM
    CODE("bye",   state = STOP);
    ///@}
    CODE("boot",  MU->clear(FIND("boot") + 1));
#if 0  /* words TODO */  
    CODE("?dup",  {});
    CODE("?do",   {});
    CODE("do",    {});
    CODE("i",     {});
    CODE("loop",  {});
    CODE("immediate", {});
    CODE("included", {});
    CODE("base",  {});
    CODE("bl",    {});
    CODE("depth", {});
    CODE("invert", {});
    CODE("is",    {});
    CODE("leave", {});
    CODE("lshift", {});
    CODE("rshift", {});
    CODE("nip",    {});
    CODE("roll",   {});
    CODE("u.",     {});
    CODE("u<",     {});
    CODE("u>",     {});
    CODE("value",  {});
    CODE("within", {});
#endif
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
ForthVM::number(char *idiom) {
    char *p;
    DU n = (STRCHR(idiom, '.'))
        ? STRTOF(idiom, &p)
        : STRTOL(idiom, &p, *base);
    if (*p != '\0') return 0;            /// * not a number, bail
    // is a number
    if (compile) {                       /// * add literal when in compile mode
        VLOG2("%d| %f\n", id, n);
        add_w(DOLIT);                    ///> dovar (+parameter field)
        add_du(n);                       ///> store literal
    }
    else {                               ///> or, add value onto data stack
        VLOG2("%d| ss.push(%f)\n", id, n);
        PUSH(n);
    }
    return 1;
}
//=======================================================================================
