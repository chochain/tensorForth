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
///    also we can change pmem implementation anytime without affecting opcodes defined below
///
///@name parameter memory load/store macros
///@{
#define PFA(w)    (dict[(IU)(w)].pfa)             /**< PFA of given word id                 */
#define HERE      (mmu->here())                   /**< current context                      */
#define MEM(a)    (mmu->pmem((IU)(a)))            /**< parameter memory by offset address   */
#define CELL(a)   (*(DU*)mmu->pmem((IU)a))        /**< fetch a cell from parameter memory   */
#define LAST      (mmu->dict(mmu->dict._didx-1))  /**< last colon word defined              */
#define BASE      ((U8*)MEM(base))                /**< pointer to user area per VM          */
#define SETJMP(a) (((Param*)MEM(a))->ioff = HERE) /**< set branch target                    */
///@}
///@name stack op macros
///@{
#define PUSH(v) (ss.push(tos), tos = v)
#define POP()   ({ DU n=tos; tos=ss.pop(); n; })
#define POPI()  (D2I(POP()))
#define SS2I    ((id<<10)|(ss.idx>=0 ? ss.idx : 0)) /**< ss_dump parameter (composite)     */
///@}

__GPU__
ForthVM::ForthVM(int id, System *sys) : VM(id, sys) {
    dict = mmu->dict(0);
    base = id;                                  /// * pmem[id], 0..USER_AREA-1 reserved
    *MEM(base) = 10;
    VLOG1("\\  ::ForthVM[%d] dict=%p\n", id, dict);
}
///
/// resume suspended task
///
__GPU__ int
ForthVM::resume() {
    VLOG1("VM[%d] resumed at ip=%x\n", id, ip);
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

__GPU__ int
ForthVM::post() {
    DEBUG("%d> VM.state=%d\n", id, state);
    if (state!=HOLD && !compile) {
        sys->op(OP_SS, *BASE, tos, SS2I);
    }
    return 0;
}
///
///@name Forth Inter-Interpreter
///@{
#define DISPATCH(op) switch(op)
#define CASE(op, g)  case op : { g; } break
#define OTHER(g)     default : { g; } break
#define UNNEST()     (ip=D2I(rs.pop()))

__GPU__ void
ForthVM::nest() {
    state = NEST;
    ///
    /// when ip != 0, it resumes paused VM
    ///
    while (ip) {                                     /// * try no recursion
        Param &ix = *(Param*)MEM(ip);
        VM_HDR(":%x", ix.op);
        ip += sizeof(IU);
        DISPATCH(ix.op) {                            /// * opcode dispatcher
        CASE(EXIT, UNNEST());
        CASE(NEXT,
             if (GT(rs[-1]-=DU1, -DU1)) {            ///> loop done?
                 ip = ix.ioff;                       /// * no, loop back
             }
             else rs.pop());                         /// * yes, loop done!
        CASE(LOOP,
             if (GT(rs[-2], rs[-1] += DU1)) {        ///> loop done?
                 ip = ix.ioff;                       /// * no, loop back
             }
             else { rs.pop(); rs.pop(); });          /// * yes, done, pop off counters
        CASE(LIT,
             ss.push(tos);                           ///> push current tos
             tos = *(DU*)MEM(ip);                    /// * fetch from next IU
             ip += sizeof(DU);                       /// * advance ip
             if (ix.exit) UNNEST());                 ///> constant/value
        CASE(VAR,
             PUSH(ALIGN(ip));                        ///> get var addr
             if (ix.ioff) ip = ix.ioff;              /// * jmp to does>
             else UNNEST());                         /// * 0: variable
        CASE(STR,
             PUSH(ip); PUSH(ix.ioff); ip += ix.ioff);
        CASE(DOTQ,                                   /// ." ..."
             const char *s = (const char*)MEM(ip);   ///< get string pointer
             sys->pstr(s); ip += ix.ioff);           /// * send to output console
        CASE(BRAN,  ip = ix.ioff);                   /// * unconditional branch
        CASE(ZBRAN, if (ZEQ(POP())) ip = ix.ioff);   /// * conditional branch
        CASE(FOR, rs.push(POP()));                   /// * setup FOR..NEXT call frame
        CASE(DO,                                     /// * setup DO..LOOP call frame
             rs.push(ss.pop()); ss.push(POP())); 
        CASE(KEY, PUSH(sys->key()); UNNEST());       /// * fetch single keypress
        OTHER(
            if (ix.udf) {                            /// * user defined word?
                rs.push(ip);                         /// * setup call frame
                ip = ix.ioff;                        /// * ip = word.pfa
            }
            else (*mmu->XT(ix.ioff))());             /// * execute built-in word
        }
        VM_TLR(" => SS=%d, RS=%d, ip=%x", ss.idx, rs.idx, ip);
    }
}
///
///> CALL - inner-interpreter proxy (inline macro does not run faster)
///
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    Code &c = dict[w];                               /// * code reference
    DEBUG(" => call(%s)\n", c.name);
    if (c.udf) {                                     /// * userd defined word
        rs.push(ip);
        ip = c.pfa;
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
    if (id != 0) return;  /// * done once only
    
    CODE("___ ",    {});  /// dict[0] not used, simplify find(), also keeps _XT0
    CODE("nop",     {});  /// do nothing
    ///
    /// @defgroup Stack ops
    /// @brief - opcode sequence can be changed below this line
    /// @{
    CODE("dup",     PUSH(tos));
    CODE("drop",    tos = ss.pop());
    CODE("over",    DU v = ss[-1]; PUSH(v));
    CODE("swap",    DU n = ss.pop(); PUSH(n));
    CODE("rot",     DU n = ss.pop(); DU m = ss.pop(); ss.push(n); PUSH(m));
    CODE("-rot",    DU n = ss.pop(); DU m = ss.pop(); PUSH(m); PUSH(n));
    CODE("pick",    IU i = D2I(tos); tos = ss[-i]);
    CODE("nip",     ss.pop());
    CODE("?dup",    if (tos != DU0) PUSH(tos));
    /// @}
    /// @defgroup Stack ops - double
    /// @{
    CODE("2dup",    DU v = ss[-1]; PUSH(v); v = ss[-1]; PUSH(v));
    CODE("2drop",   ss.pop(); tos = ss.pop());
    CODE("2over",   DU v = ss[-3]; PUSH(v); v = ss[-3]; PUSH(v));
    CODE("2swap",   DU n = ss.pop(); DU m = ss.pop(); DU l = ss.pop();
                    ss.push(n); PUSH(l); PUSH(m));
    /// @}
    /// @defgroup ALU ops
    /// @{
    CODE("+",       tos += ss.pop());
    CODE("*",       tos *= ss.pop());
    CODE("-",       tos =  ss.pop() - tos);
    CODE("/",       tos =  ss.pop() / tos);
    CODE("mod",     tos =  INT(MOD(ss.pop(), tos)));            // ( a b -- c )   c=int(a%b)
    CODE("fmod",    tos =  MOD(ss.pop(), tos));                 // ( a b -- c )   c=a%b      
    CODE("*/",      tos =  MUL2(ss.pop(), ss.pop()) / tos);     // ( a b c -- d ) d= a*b / c 
    CODE("/mod",    DU  n = ss.pop();                           // ( a b -- c d ) c=a%b, d=a/b
                    DU  t = tos;
                    DU  m = MOD(n, t);
                    ss.push(m); tos = INT(n / t));
    CODE("*/mod",   DU2 n = MUL2(ss.pop(), ss.pop());           // ( a b c -- d e ) d=(a*b)%c, e=(a*b)/c
                    DU2 t = tos;
                    DU  m = MOD2(n, t);
                    ss.push(m); tos = INT(n / t));
    CODE("and",     tos = D2I(tos) & D2I(ss.pop()));
    CODE("or",      tos = D2I(tos) | D2I(ss.pop()));
    CODE("xor",     tos = D2I(tos) ^ D2I(ss.pop()));
    CODE("abs",     tos = ABS(tos));
    CODE("negate",  tos = -tos);
    CODE("invert",  tos = ~D2I(tos));
    CODE("rshift",  tos = D2I(ss.pop()) >> D2I(tos));
    CODE("lshift",  tos = D2I(ss.pop()) << D2I(tos));
    CODE("max",     DU n=ss.pop(); tos = (tos>n) ? tos : n);
    CODE("min",     DU n=ss.pop(); tos = (tos<n) ? tos : n);
    CODE("2*",      tos *= 2);
    CODE("2/",      tos /= 2);
    CODE("1+",      tos += 1);
    CODE("1-",      tos -= 1);
    /// @}
    /// @defgroup Data conversion ops
    /// @{
    CODE("f>s",     tos = INT(tos));     /// nearest-even 0.5 => 0, 1.5 => 2, 2.5 => 2
    CODE("round",   tos = round(tos));   /// 0.5 => 1, 1.5 => 2, 2.5 => 3, 1.5 => -2 
    CODE("ceil",    tos = ceilf(tos));   /// 1.5 => 2, -1.5 => -1
    CODE("floor",   tos = floorf(tos));  /// 1.5 => 1, -1.5 => -2
    ///@}
    /// @defgroup Logic ops
    /// @{
    CODE("0=",      tos = BOOL(ZEQ(tos)));
    CODE("0<",      tos = BOOL(LT(tos, DU0)));
    CODE("0>",      tos = BOOL(GT(tos, DU0)));
    CODE("=",       tos = BOOL(EQ(ss.pop(), tos)));
    CODE(">",       tos = BOOL(GT(ss.pop(), tos)));
    CODE("<",       tos = BOOL(LT(ss.pop(), tos)));
    CODE("<>",      tos = BOOL(!EQ(ss.pop(), tos)));
    CODE(">=",      tos = BOOL(!LT(ss.pop(), tos)));
    CODE("<=",      tos = BOOL(!GT(ss.pop(), tos)));
    CODE("u<",      tos = BOOL(UINT(D2I(ss.pop())) < UINT(D2I(tos))));
    CODE("u>",      tos = BOOL(UINT(D2I(ss.pop())) > UINT(D2I(tos))));
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
    CODE("i",       PUSH(rs[-1]));
    CODE("leave",   rs.pop(); rs.pop(); UNNEST());         // quit DO..LOOP
    IMMD("loop",    add_p(LOOP, POPI()));                  // next ( here -- )
    /// @}
    /// @defgrouop return stack ops
    /// @{
    CODE(">r",      rs.push(POP()));
    CODE("r>",      PUSH(rs.pop()));
    CODE("r@",      PUSH(rs[-1]));                              // same as I (the loop counter)
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
         add_p(BRAN, ip); UNNEST());                            // jmp to next ip
    IMMD("to", _to_value());                                    // alter the value of a constant, i.e. 3 to x
    IMMD("is", _is_alias());                                    // alias a word, i.e. ' y is x
/*    
    CODE("[to]",            // : xx 3 [to] y ;                  // alter constant in compile mode
         IU w = LDi(ip); ip += sizeof(IU);                      // fetch constant pfa from 'here'
         IU a = PFA(w) + sizeof(IU);
         DU d = POP();
         if (a < T4_PMEM_SZ) CELL(a) = d;                       // store tos into constant pfa
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
    CODE("th",    IU i = POPI(); tos += i * sizeof(DU));        // w i -- w'
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
    CODE("abort", tos = -DU1; ss.clear(); rs.clear());          // clear ss, rs
    CODE("here",  PUSH(HERE));
    CODE("'",     IU w = FIND(sys->fetch()); if (w) PUSH(w));
    CODE(".s",    sys->op(OP_SS, *BASE, tos, SS2I));
    CODE("depth", PUSH(ss.idx - 1));
    CODE("words", sys->op(OP_WORDS));
    CODE("dict",  sys->op(OP_DICT));                            // dict_dump in host mode
    CODE("dict_dump", mmu->dict_dump());
    CODE("see",   IU w = FIND(sys->fetch()); if (!w) return;
                  sys->op(OP_SEE, *BASE, DU0, w));
    CODE("dump",  DU n = POP(); IU a = POPI();
                  sys->op(OP_DUMP, 0, n, a));
    CODE("forget", _forget());
    CODE("trace", sys->trace(POPI()));                          // set debug/trace level
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
    CODE("power",  {}); // power(2, 3)
    CODE("?do",    {});
    CODE("roll",   {});
    CODE("u<",     {});
    CODE("u>",     {});
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
    IU w = FIND(idiom);                   /// * search through dictionary
    if (!w) {                             /// * input word not found
        DEBUG(" '%s' not found\n", idiom);
        return 0;                         /// * next, try as a number
    }
    Code &c = dict[w];
    DEBUG("%04x[%3x]%c%c %s",
          c.udf ? c.pfa : mmu->XTOFF(c.xt), w,
          c.imm ? '*' : ' ', c.udf ? 'u' : ' ',
          c.name);
    if (compile && !c.imm) {              /// * in compile mode?
        add_w((IU)w);                     /// * add found word to new colon word
    }
    else { ip = DU0; call((IU)w); }       /// * execute forth word
    
    return 1;
}
///
/// parse input idiom as a number
///
__GPU__ int
ForthVM::number(char *idiom) {
    int b = *BASE;
    switch (*idiom) {                     ///> base override
    case '%': b = 2;  idiom++; break;
    case '&':
    case '#': b = 10; idiom++; break;
    case '$': b = 16; idiom++; break;
    }
    char *p;
    DU2 n = (b==10 && STRCHR(idiom, '.'))
        ? STRTOF(idiom, &p)
        : STRTOL(idiom, &p, b);
    if (*p != '\0') {                     /// * not a number, bail
        DEBUG(" number(%s) base=%d => error\n", idiom, b);
        return 0;
    }
    // is a number
#if T4_VERBOSE > 1    
    DU m = (DU)n;
    p = (char*)&m;
    for (int i=0; i<sizeof(DU); i++, p++) {
        const char h2c[] = "0123456789abcdef";
        DEBUG("%c%c ", h2c[((*p)>>4)&0xf], h2c[(*p)&0xf]);
    }
#endif // T4_VERBOSE > 1
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
    IU w = FIND(name);
    DEBUG("_def_word(%s) => %d\n", name, w);
    if (w) {                              /// * word redefined?
        sys->pstr(name);
        sys->pstr(" reDef? ", CR);
    }
    mmu->colon(name);                     /// * create a colon word
    return 1;                             /// * created OK
}
__GPU__ void
ForthVM::_forget() {
    IU w = FIND(sys->fetch()); if (!w) return; /// bail, if not found
    IU b = FIND((char*)"boot")+1;
    mmu->clear(w > b ? w : b);
}
__GPU__ void
ForthVM::_quote(prim_op op) {
    const char *s = sys->scan('"')+1;     ///> string skip first blank
    if (compile) {
        add_p(op, ALIGN(STRLEN(s)+1));    ///> dostr, (+parameter field)
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
