/** -*- c++ -*-
 * @file
 * @brief ForthVM class - eForth VM implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "eforth.h"
///
/// Forth Virtual Machine operational macros to reduce verbosity
/// Note:
///    so we can change pmem implementation anytime without affecting opcodes defined below
///
///@name Dictioanry access
///@{
#define PFA(w)    (dict[(IU)(w)].pfa)           /**< PFA of given word id                 */
#define HERE      (mmu->here())                 /**< current context                      */
#define SETJMP(a) (mmu->setjmp(a))              /**< address offset for branching opcodes */
///@}
///@name Heap memory load/store macros
///@{
#define MEM(a)    (mmu->pmem((IU)(a)))
#define CELL(a)   (*(DU*)mmu->pmem((IU)a))      /**< fetch a cell from parameter memory    */
#define LAST      (mmu->dict(mmu->dict._didx-1))/**< last colon word defined               */
#define BASE      ((int*)MEM(base))
//#define LDi(a)    (mmu->ri((IU)(a)))            /**< read an instruction unit from pmem    */
//#define LDd(a)    (mmu->rd((IU)(a)))            /**< read a data unit from pmem            */
//#define STi(a,d)  (mmu->wi((IU)(a), (IU)(d)))   /**< write a instruction unit to pmem      */
//#define STd(a,d)  (mmu->wd((IU)(a), (DU)(d)))   /**< write a data unit to pmem             */
//#define STc(a,c)  (*((char*)MEM(w))=(U8)INT(c)) /**< write a char to pmem                  */
///@}
///@name stack op macros
///@{
#define PUSH(v) (SS.push(TOS), TOS = v)
#define POP()   ({ DU n=TOS; TOS=SS.pop(); n; })
#define POPI()  (UINT(POP()))
///@}

__GPU__
ForthVM::ForthVM(int id, System *sys) : VM(id, sys) {
    dict = mmu->dict(0);
    base = id;                                  /// * pmem[id], 0..USER_AREA-1 reserved
    VLOG1("\\  ::ForthVM[%d] dict=%p\n", id, dict);
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
__GPU__ int
ForthVM::process(char *idiom) {
    return parse(idiom) || number(idiom);
}
/*
__GPU__ int
ForthVM::post() {
    cudaError_t code = cudaGetLastError();
    if (code == cudaSuccess) return 0;
    
    ERROR("VM ERROR: %s %s %d, IP=%x, CELL(IP)=%f\n",
          cudaGetErrorString(code), __FILE__, __LINE__, IP, CELL(IP));
    return 1;
}
*/
#define VM_HDR(fmt, ...)                      \
    printf("\e[%dm[%02d.%d]%-4x" fmt "\e[0m", \
           (id&7) ? 38-(id&7) : 37, id, state, IP, ##__VA_ARGS__)
#define VM_TLR(fmt, ...)                      \
    printf("\e[%dm" fmt "\e[0m\n",            \
           (id&7) ? 38-(id&7) : 37, ##__VA_ARGS__)
#define VM_LOG(fmt, ...)                      \
    VM_HDR(fmt, ##__VA_ARGS__);               \
    printf("\n")
///
///@name Forth Inter-Interpreter
///@{
#define DISPATCH(op) switch(op)
#define CASE(op, g)  case op : { g; } break
#define OTHER(g)     default : { g; } break
#define UNNEST()     (IP=INT(RS.pop()))

__GPU__ void
ForthVM::nest() {
    state = NEST;
    ///
    /// when IP != 0, it resumes paused VM
    ///
    while (IP) {                                     /// * try no recursion
        Param &ix = *(Param*)MEM(IP);
        IP += sizeof(IU);
        VM_HDR(":%x", ix.op);
        IP += sizeof(IU);
        DISPATCH(ix.op) {                            /// * opcode dispatcher
        CASE(EXIT, UNNEST());
        CASE(NEXT,
             if (GT(RS[-1]-=DU1, -DU1)) {            ///> loop done?
                 IP = ix.ioff;                       /// * no, loop back
             }
             else RS.pop());                         /// * yes, loop done!
        CASE(LOOP,
             if (GT(RS[-2], RS[-1] += DU1)) {        ///> loop done?
                 IP = ix.ioff;                       /// * no, loop back
             }
             else { RS.pop(); RS.pop(); });          /// * yes, done, pop off counters
        CASE(LIT,
             SS.push(TOS);                           ///> push current TOS
             TOS = *(DU*)MEM(IP);                    /// * fetch from next IU
             IP += sizeof(DU);                       /// * advance IP
             if (ix.exit) UNNEST());                 ///> constant/value
        CASE(VAR,
             PUSH(ALIGN(IP));                        ///> get var addr
             if (ix.ioff) IP = ix.ioff;              /// * jmp to does>
             else UNNEST());                         /// * 0: variable
        CASE(STR,
             PUSH(IP); PUSH(ix.ioff); IP += ix.ioff);
        CASE(DOTQ,                                   /// ." ..."
             const char *s = (const char*)MEM(IP);   ///< get string pointer
             sys->pstr(s); IP += ix.ioff);           /// * send to output console
        CASE(BRAN,  IP = ix.ioff);                   /// * unconditional branch
        CASE(ZBRAN, if (ZEQ(POP())) IP = ix.ioff);   /// * conditional branch
        CASE(FOR, RS.push(POPI()));                  /// * setup FOR..NEXT call frame
        CASE(DO,                                     /// * setup DO..LOOP call frame
             RS.push(SS.pop()); SS.push(POPI())); 
        CASE(KEY, PUSH(sys->key()); UNNEST());       /// * fetch single keypress
        OTHER(
            if (ix.udf) {                            /// * user defined word?
                RS.push(IP);                         /// * setup call frame
                IP = ix.ioff;                        /// * IP = word.pfa
            }
            else (*mmu->XT(ix.ioff))());             /// * execute built-in word
        }
        VM_TLR(" => SS=%d, RS=%d, IP=%x", SS.idx, RS.idx, IP);
    }
}
///
///> CALL - inner-interpreter proxy (inline macro does not run faster)
///
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    Code &c = dict[w];                               /// * code reference
    DEBUG(" => call(%s)\n", c.name);
    if (c.udf) {                                     /// * userd defined word
        RS.push(IP);
        IP = c.pfa;
        nest();                                      /// * Forth inner loop
    }
    else (*(FPTR)((UFP)c.xt & MSK_XT))();            /// * execute function
}
///
/// dictionary initializer
///
__GPU__ void
ForthVM::init() {
    VM::init();
    if (id != 0) return;       /// * done once only
    CODE("nul ",    {});               /// dict[0], not used, simplify find()
    CODE("nop",     {});               /// do nothing
    ///
    /// @defgroup Stack ops
    /// @brief - opcode sequence can be changed below this line
    /// @{
    CODE("dup",     PUSH(TOS));
    CODE("drop",    TOS = SS.pop());
    CODE("over",    DU v = SS[-1]; PUSH(v));
    CODE("swap",    DU n = SS.pop(); PUSH(n));
    CODE("rot",     DU n = SS.pop(); DU m = SS.pop(); SS.push(n); PUSH(m));
    CODE("-rot",    DU n = SS.pop(); DU m = SS.pop(); PUSH(m); PUSH(n));
    CODE("pick",    IU i = UINT(TOS); TOS = SS[-i]);
    CODE("nip",     SS.pop());
    CODE("?dup",    if (TOS != DU0) PUSH(TOS));
    /// @}
    /// @defgroup Stack ops - double
    /// @{
    CODE("2dup",    DU v = SS[-1]; PUSH(v); v = SS[-1]; PUSH(v));
    CODE("2drop",   SS.pop(); TOS = SS.pop());
    CODE("2over",   DU v = SS[-3]; PUSH(v); v = SS[-3]; PUSH(v));
    CODE("2swap",   DU n = SS.pop(); DU m = SS.pop(); DU l = SS.pop();
                    SS.push(n); PUSH(l); PUSH(m));
    /// @}
    /// @defgroup ALU ops
    /// @{
    CODE("+",       TOS += SS.pop());
    CODE("*",       TOS *= SS.pop());
    CODE("-",       TOS =  SS.pop() - TOS);
    CODE("/",       TOS =  SS.pop() / TOS);
    CODE("mod",     TOS =  MOD(SS.pop(), TOS));
    CODE("*/",      TOS =  (DU2)SS.pop() * SS.pop() / TOS);     // ( a b c --- d ) a * b / c 
    CODE("/mod",    DU  n = SS.pop();
                    DU  t = TOS;
                    DU  m = MOD(n, t);
                    SS.push(m); TOS = UINT(n / t));
    CODE("*/mod",   DU2 n = (DU2)SS.pop() * SS.pop();
                    DU2 t = TOS;
                    DU  m = MOD(n, t);
                    SS.push(m); TOS = UINT(n / t));
    CODE("and",     TOS = UINT(TOS) & UINT(SS.pop()));
    CODE("or",      TOS = UINT(TOS) | UINT(SS.pop()));
    CODE("xor",     TOS = UINT(TOS) ^ UINT(SS.pop()));
    CODE("abs",     TOS = ABS(TOS));
    CODE("negate",  TOS = -TOS);
    CODE("invert",  TOS = ~UINT(TOS));
    CODE("rshift",  TOS = UINT(SS.pop()) >> UINT(TOS));
    CODE("lshift",  TOS = UINT(SS.pop()) << UINT(TOS));
    CODE("max",     DU n=SS.pop(); TOS = (TOS>n) ? TOS : n);
    CODE("min",     DU n=SS.pop(); TOS = (TOS<n) ? TOS : n);
    CODE("2*",      TOS *= 2);
    CODE("2/",      TOS /= 2);
    CODE("1+",      TOS += 1);
    CODE("1-",      TOS -= 1);
    /// @}
    /// @defgroup Data conversion ops
    /// @{
    CODE("int",     TOS = INT(TOS));    /// nearest-even 0.5 => 0, 1.5 => 2, 2.5 => 2
    CODE("round",   TOS = round(TOS));  /// 0.5 => 1, 1.5 => 2, 2.5 => 3, 1.5 => -2 
    CODE("ceil",    TOS = ceil(TOS));   /// 1.5 => 2, -1.5 => -1
    CODE("floor",   TOS = floor(TOS));  /// 1.5 => 1, -1.5 => -2
    ///@}
    /// @defgroup Logic ops
    /// @{
    CODE("0=",      TOS = BOOL(ZEQ(TOS)));
    CODE("0<",      TOS = BOOL(LT(TOS, DU0)));
    CODE("0>",      TOS = BOOL(GT(TOS, DU0)));
    CODE("=",       TOS = BOOL(EQ(SS.pop(), TOS)));
    CODE(">",       TOS = BOOL(GT(SS.pop(), TOS)));
    CODE("<",       TOS = BOOL(LT(SS.pop(), TOS)));
    CODE("<>",      TOS = BOOL(!EQ(SS.pop(), TOS)));
    CODE(">=",      TOS = BOOL(!LT(SS.pop(), TOS)));
    CODE("<=",      TOS = BOOL(!GT(SS.pop(), TOS)));
    CODE("u<",      TOS = BOOL(UINT(SS.pop()) < UINT(TOS)));
    CODE("u>",      TOS = BOOL(UINT(SS.pop()) > UINT(TOS)));
    /// @}
    /// @defgroup IO ops
    /// @{
    CODE("base",    PUSH(base));
    CODE("decimal", sys->dot(RDX, *BASE=10));
    CODE("hex",     sys->dot(RDX, *BASE=16));
    CODE("bl",      PUSH(0x20));
    CODE("cr",      sys->dot(CR));
    CODE(".",       sys->dot(DOT,  POP()));
    CODE("u.",      sys->dot(UDOT, POP()));
    CODE(".r",      IU i = POPI(); sys->dotr(i, POP(), *BASE));
    CODE("u.r",     IU i = POPI(); sys->dotr(i, POP(), *BASE, true));
    CODE("type",    POP(); sys->pstr((const char*)MEM(POPI())));     // pass string pointer
    IMMD("key",     if (compile) add_p(KEY); else PUSH(sys->key()));
    CODE("emit",    sys->dot(EMIT, POP()));
    CODE("space",   sys->dot(SPCS, DU1));
    CODE("spaces",  sys->dot(SPCS, POP()));
    /// @}
    /// @defgroup Literal ops
    /// @{
    IMMD("(",       sys->scan(')'));
    IMMD(".(",      sys->pstr(sys->scan(')')));
    IMMD("\\",      sys->scan('\n'));
    IMMD("s\"",     _quote(STR));
    IMMD(".\"",     _quote(DOTQ));
    /// @}
    /// @defgroup Branching ops
    /// @brief - if...then, if...else...then
    /// @{
    IMMD("if",      PUSH(HERE); add_p(ZBRAN));             // if    ( -- here )
    IMMD("else",    IU h=HERE;  add_p(BRAN);               // else ( here -- there )
                    SETJMP(POPI()); PUSH(h));
    IMMD("then",    SETJMP(POPI()));                       // backfill jump address
    /// @}
    /// @defgroup Loops
    /// @brief  - begin...again, begin...f until, begin...f while...repeat
    /// @{
    IMMD("begin",   PUSH(HERE));
    IMMD("again",   add_p(BRAN, POPI()));                  // again    ( there -- )
    IMMD("until",   add_p(ZBRAN, POPI()));                 // until    ( there -- )
    IMMD("while",   PUSH(HERE); add_p(ZBRAN));             // while    ( there -- there here )
    IMMD("repeat",                                         // repeat    ( there1 there2 -- )
         IU t=POPI(); add_p(BRAN, POPI()); SETJMP(t));     // set forward and loop back address
    /// @}
    /// @defgrouop FOR...NEXT loops
    /// @brief  - for...next, for...aft...then...next
    ///    3 for ." f" aft ." a" then i . next  ==> f3 a2 a1 a0 i.e. f once only
    /// @{
    IMMD("for" ,    add_p(FOR); PUSH(HERE));               // for ( -- here )
    IMMD("next",    add_p(NEXT, POPI()));                  // next ( here -- )
    IMMD("aft",                                            // aft ( here -- here there )
         POP(); IU h=HERE; add_p(BRAN); PUSH(HERE); PUSH(h));
    /// @}
    /// @}
    /// @defgrouop DO..LOOP loops
    /// @{
    IMMD("do" ,     add_p(DO); PUSH(HERE));                // do ( -- here )
    CODE("i",       PUSH(RS[-1]));
    CODE("leave",   RS.pop(); RS.pop(); UNNEST());         // quit DO..LOOP
    IMMD("loop",    add_p(LOOP, POPI()));                  // next ( here -- )
    /// @}
    /// @defgrouop return stack ops
    /// @{
    CODE(">r",      RS.push(POP()));
    CODE("r>",      PUSH(RS.pop()));
    CODE("r@",      PUSH(RS[-1]));                              // same as I (the loop counter)
    /// @}
    /// @defgrouop Compiler ops
    /// @{
    CODE("[",       compile = false);
    CODE("]",       compile = true);
    CODE(":",       compile = _def_word());
    IMMD(";",       add_p(EXIT); compile = false);
    CODE("variable",                                            // create a variable
         if (!_def_word()) return;
         add_p(VAR, 0, true); add_du(DU0));                     // default DU0
    CODE("constant",                                            // create a constant
         if (!_def_word()) return;
         add_lit(POP(), true));
    CODE("value",   
         if (!_def_word()) return;
         add_p(LIT, 0, true, true);                             // forced extended, TO can update
         add_du(POP()));             
    IMMD("immediate", dict[-1].imm = true);
    CODE("exit",    UNNEST());                                  // early exit the colon word
    /// @}
    /// @defgroup metacompiler
    /// @brief - dict is directly used, instead of shield by macros
    /// @{
    CODE("exec",   IU w = POP(); call(w));                      // execute word
    CODE("create",
         if (!_def_word()) return;
         add_p(VAR, 0, true));
    CODE("does>",
         IU pfa = mmu->last()->pfa;
         while (((Param*)MEM(pfa))->op != VAR && (pfa < (IU)HERE)) {  // find that VAR
             pfa += sizeof(IU);
         }
         SETJMP(pfa);                                           // set jmp target
         add_p(BRAN, IP); UNNEST());                            // jmp to next IP
    IMMD("to", _to_value());                                    // alter the value of a constant, i.e. 3 to x
    IMMD("is", _is_alias());                                    // alias a word, i.e. ' y is x
/*    
    CODE("[to]",            // : xx 3 [to] y ;                  // alter constant in compile mode
         IU w = LDi(IP); IP += sizeof(IU);                      // fetch constant pfa from 'here'
         IU a = PFA(w) + sizeof(IU);
         DU d = POP();
         if (a < T4_PMEM_SZ) CELL(a) = d;                       // store TOS into constant pfa
         else { ERROR("is %x", a); state = STOP; });
*/         
    ///
    /// be careful with memory access, because
    /// it could make access misaligned which cause exception
    ///
    CODE("@",     IU i = POPI(); PUSH((DU)CELL(i)));            // i -- n
    CODE("!",     IU i = POPI(); CELL(i) = POP(););             // n i --
    CODE("+!",    IU i = POPI(); CELL(i) += POP());             // n i --
    CODE("?",     IU i = POPI(); sys->dot(DOT, CELL(i)));       // i --
    CODE(",",     DU n = POP();  add_du(n));                    // n -- , compile a cell
    CODE("cells", IU i = POPI(); PUSH(i * sizeof(DU)));         // n -- n'
    CODE("allot",                                               // n --
         IU n = POPI();                                         // number of bytes
         for (IU i = 0; i < n; i+=sizeof(DU)) add_du(DU0));     // zero padding
    CODE("th",    IU i = POPI(); TOS += i * sizeof(DU));        // w i -- w'
    /// @}
#if DO_MULTITASK    
    /// @defgroup Multitasking ops
    /// @}
    CODE("task",                                                // w -- task_id
         IU i = POPI(); Code &c = dict[i];                      ///< dictionary index
         if (c.udf) PUSH(task_create(c.pfa));                   /// create a task starting on pfa
         else pstr("  ?colon word only\n"));
    CODE("rank",  PUSH(id));                                    /// ( -- task_id ) used insided a task
    CODE("start", task_start(POPI()));                          /// ( task_id -- )
    CODE("join",  join(POPI()));                                /// ( task_id -- )
    CODE("lock",  io_lock());                                   /// wait for IO semaphore
    CODE("unlock",io_unlock());                                 /// release IO semaphore
    CODE("send",  IU t = POPI(); send(t, POPI()));              /// ( v1 v2 .. vn n tid -- ) pass values onto task's stack
    CODE("recv",  recv());                                      /// ( -- v1 v2 .. vn ) waiting for values passed by sender
    CODE("bcast", bcast(POPI()));                               /// ( v1 v2 .. vn -- )
    CODE("pull",  IU t = POPI(); pull(t, POPI()));              /// ( n task_id -- v1 v2 .. vn )
    /// @}
#endif // DO_MULTITASK    
    /// @defgroup Debug ops
    /// @{
    CODE("abort", TOS = -DU1; SS.clear(); RS.clear());          // clear ss, rs
    CODE("here",  PUSH(HERE));
    CODE("'",     int w = FIND(sys->fetch()); PUSH(w));
    CODE(".s",    sys->op(OP_SS, id));
    CODE("words", sys->op(OP_WORDS));
    CODE("see",   int w = FIND(sys->fetch()); sys->op(OP_SEE, w));
    CODE("dump",  DU n = POP(); int a = POPi; sys->op(OP_DUMP, a, n));
    CODE("forget", _forget());
    /// @}
    /// @defgroup OS ops
    /// @{
    CODE("mstat", mmu->status());
    CODE("rnd",   PUSH(sys->rand(DU1, NORMAL)));                // generate random number
    CODE("ms",    delay(POPI()));
//    CODE("included",                                          // include external file
//         POP();                                               // string length, not used
//         sys->load(MEM(POP())));                              // include external file
    CODE("clock", DU t = sys->ms(); SCALAR(t); PUSH(t));
    CODE("bye",   state = STOP);                                // atomicExch(&state, STOP)
    ///@}
    CODE("boot",  mmu->clear(FIND((char*)"boot") + 1));
#if 0  /* words TODO */
    CODE("^",     {}); // power(2, 3)
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
    VLOG1("ForthVM[%d]::init ok\n", id);
};
///======================================================================
///
/// parse input idiom as a word
///
__GPU__ int
ForthVM::parse(char *idiom) {
    state = QUERY;
    int w = FIND(idiom);                  /// * search through dictionary
    if (w < 0) {                          /// * input word not found
        DEBUG(" '%s' not found\n", idiom);
        return 0;                         /// * next, try as a number
    }
    Code &c = dict[w];
    DEBUG(" %c%c %06x: %s %d",
          c.imm ? 'I' : ' ', c.udf ? 'U' : ' ',
          c.udf ? c.pfa : mmu->XTOFF(c.xt), c.name, w);
    if (compile && !c.imm) {              /// * in compile mode?
        add_w((IU)w);                     /// * add found word to new colon word
    }
    else { IP = DU0; call((IU)w); }       /// * execute forth word
    
    return 1;
}
///
/// parse input idiom as a number
///
__GPU__ int
ForthVM::number(char *idiom) {
    int base;
    switch (*idiom) {                     ///> base override
    case '%': base = 2;  idiom++; break;
    case '&':
    case '#': base = 10; idiom++; break;
    case '$': base = 16; idiom++; break;
    default:  base = 10;
    }
    char *p;
    DU n = (STRCHR(idiom, '.'))
        ? STRTOF(idiom, &p)
        : STRTOL(idiom, &p, base);
    if (*p != '\0') return 0;            /// * not a number, bail
    
    // is a number
    if (compile) add_lit((DU)n);          /// * add literal when in compile mode
    else         PUSH((DU)n);             ///> or, add value onto data stack
    
    return 1;
}
///
///@name misc eForth functions (in Standard::Core section)
///@{
__GPU__ int
ForthVM::_def_word() {                    ///< display if redefined
    char *name = sys->fetch();
    if (name[0]=='\0') {                  /// * missing name?
        sys->pstr(" name?", CR); return 0;
    }  
    if (FIND(name)) {                     /// * word redefined?
        sys->pstr(name);
        sys->pstr(" reDef? ", CR);
    }
    mmu->colon(name);                     /// * create a colon word
    return 1;                             /// * created OK
}
__GPU__ void
ForthVM::_forget() {
    int w = FIND(sys->fetch()); if (!w) return;             // bail, if not found
    int b = FIND((char*)"boot")+1;
    mmu->clear(w > b ? w : b);
}
__GPU__ void
ForthVM::_quote(prim_op op) {
    const char *s = sys->scan('"')+1;     ///> string skip first blank
    if (compile) {
        add_p(op, STRLEN(s));             ///> dostr, (+parameter field)
        add_str(s);                       ///> byte0, byte1, byte2, ..., byteN
    }
    else {
        IU h0  = HERE;                    ///> keep current memory addr
        DU len = add_str(s);              ///> write string to PAD
        switch (op) {
        case STR:  PUSH(h0); PUSH(len);             break; ///> addr, len
        case DOTQ: sys->pstr((const char*)MEM(h0)); break; ///> to console
        default:   sys->pstr("_quote unknown op:");
        }
        mmu->set_here(h0);                ///> restore memory addr
    }
}
__GPU__ void
ForthVM::_to_value() {                    ///> update a constant/value
    IU w = state==QUERY ? FIND(sys->fetch()) : POPI();     // constant addr
    if (!w) return;
    if (compile) {
        add_lit((DU)w);                                    // save addr on stack
        add_w(FIND((char*)"to"));                          // encode to opcode
    }
    else {
        U8    *pfa = MEM(dict[w].pfa);                     // fetch constant pointer
        Param &p   = *(Param*)(pfa);
        if (p.op==LIT) {
            *(DU*)(pfa + sizeof(IU)) = POP();              // update constant value
        }
    }
}
__GPU__ void
ForthVM::_is_alias() {                                     // create alias function
    IU w = state==QUERY ? FIND(sys->fetch()) : POPI();     // word addr
    if (!w) return;
    if (compile) {
        add_lit((DU)w);                                    // save addr on stack
        add_w(FIND((char*)"is"));
    }
    else dict[POPI()].xt = dict[w].xt;
}
///@}
//=======================================================================================
