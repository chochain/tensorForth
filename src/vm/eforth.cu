/** -*- c++ -*-
 * @file
 * @brief ForthVM class - eForth VM implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "eforth.h"

#if (T4_DO_OBJ && T4_DO_NN)
#include "mu/dataset.h"
#include "nn/model.h"
#endif // (T4_DO_OBJ && T4_DO_NN)

namespace t4::vm {
using mu::Code;
using mu::Dataset;
using nn::Model;

__HOST__
ForthVM::ForthVM(int id, System &sys) : VM(id, sys) {
    dict = mmu.dict(0);
    base = id;                                 /// * pmem[id], 0..USER_AREA-1 reserved
    *MEM(base) = 10;
    TRACE("\\  ::ForthVM[%d]\n", id);
}
///
/// resume suspended task
///
__HOST__ void
ForthVM::resume() {
    DEBUG("VM[%d] resumed at ip=%x\n", id, ip);
    nest();                                    /// * will set state to VM_READY
    post();
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
__HOST__ int
ForthVM::process(char *idiom) {
    state = QUERY;
    IU w = parse(idiom);                      /// * parse it as a word
    if (w) return 1;                          /// * success, done
    
    char *p;
    DU n = number(idiom, &p);                 /// * parse it as numeral/literal
    if (*p!='\0') return 0;                   /// * failed, bail
    
    if (compile) add_lit(n);                  /// * add literal when in compile mode
    else         PUSH(n);                     ///> or, add value onto data stack
    
    return 1;                                 /// * success
}

__HOST__ int
ForthVM::post() {
    DEBUG("vm%d> VM.state=%d\n", id, state);
    if (state!=HOLD && !compile) _ss_dump();
    return 0;
}
///
///@name Forth Inter-Interpreter
///@{
#define DISPATCH(op) switch(op)
#define CASE(op, g)  case op : { g; } break
#define OTHER(g)     default : { g; } break
#define UNNEST()     (ip=D2I(rs.pop()))

__HOST__ void
ForthVM::nest() {
    state = NEST;
    ///
    /// when ip != 0, it resumes paused VM
    ///
    while (ip && state==NEST) {                      /// * try no recursion
        Param &ix = *(Param*)MEM(ip);
        VM_HDR(":%x", ix.op);
        ip += sizeof(IU);
        DISPATCH(ix.op) {                            /// * opcode dispatcher
        CASE(EXIT, UNNEST());
        CASE(NEXT,
#if (T4_DO_OBJ && T4_DO_NN)
             if (IS_OBJ(tos) && IS_OBJ(rs[-1])) {
                 _ds_next(ix.ioff);
             }
             else
#endif // (T4_DO_OBJ && T4_DO_NN)                 
             if (GT(rs[-1]-=DU1, -DU1)) {            ///> loop done?
                 ip = ix.ioff;                       /// * no, loop back
             }
             else rs.pop());                         /// * yes, loop done
        CASE(LOOP,
             if (GT(rs[-2], rs[-1] += DU1)) {        ///> loop done?
                 ip = ix.ioff;                       /// * no, loop back
             }
             else { rs.pop(); rs.pop(); });          /// * yes, done, pop off counters
        CASE(LIT,
             ss.push(tos);                           ///> push current tos
             tos = DUP(*(DU*)MEM(ip));               /// * fetch from next IU
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
             sys.pstr(s); ip += ix.ioff);            /// * send to output console
        CASE(BRAN,  ip = ix.ioff);                   /// * unconditional branch
        CASE(ZBRAN, if (ZEQ(POP())) ip = ix.ioff);   /// * conditional branch
        CASE(FOR, rs.push(POP()));                   /// * setup FOR..NEXT call frame
        CASE(DO,                                     /// * setup DO..LOOP call frame
             rs.push(ss.pop()); rs.push(POP())); 
        CASE(KEY, PUSH(sys.key()); state=HOLD);      /// * fetch single keypress
        OTHER(
            if (ix.udf) {                            /// * user defined word?
                rs.push(ip);                         /// * setup call frame
                ip = ix.ioff;                        /// * ip = word.pfa
            }
            else (*mmu.XT(ix.ioff))());              /// * execute built-in word
        }
        VM_TLR(" => SS=%d, RS=%d, ip=%x", ss.idx, rs.idx, ip);
    }
}
///
///> CALL - inner-interpreter proxy (inline macro does not run faster)
///
__HOST__ __INLINE__ void ForthVM::call(IU w) {
    using mu::FPTR;
    using mu::MSK_ATTR;
    
    Code &c = dict[w];                               /// * code reference
    DEBUG(" => call(%s) state=%d {\n", c.name, state);
    if (c.udf) {                                     /// * userd defined word
        rs.push(ip);
        ip = c.pfa;
        nest();                                      /// * Forth inner loop
    }
    else (*(FPTR)((UFP)c.xt & MSK_ATTR))();          /// * execute function
    DEBUG("} call(%s) state=%d\n", c.name, state);
}
///
/// dictionary initializer
///
__HOST__ void
ForthVM::init() {
    VM::init();
    if (id != 0) return;    /// * done once only
    
    CODE("\nForth::", {});  /// dict[0] not used, simplify find(), also keeps _XT0
    CODE("nop",       {});  /// do nothing
    ///
    /// @defgroup Stack ops
    /// @brief - opcode sequence can be changed below this line
    /// @{
    CODE("dup",     PUSH(DUP(tos)));          ///< or create a view of a tensor (alias 'view')
    CODE("drop",    DROP(tos); tos = ss.pop());
    CODE("over",    DU v = DUP(ss[-1]); PUSH(v));
    CODE("swap",    DU n = ss.pop(); PUSH(n));
    CODE("rot",     DU n = ss.pop(); DU m = ss.pop(); ss.push(n); PUSH(m));
    CODE("-rot",    DU n = ss.pop(); DU m = ss.pop(); PUSH(m); PUSH(n));
    CODE("pick",    IU i = D2I(tos); tos = DUP(ss[-i]));
    CODE("nip",     ss.pop());
    CODE("?dup",    if (tos != DU0) PUSH(tos));
    /// @}
    /// @defgroup Stack ops - double
    /// @{
    CODE("2dup",    DU v = DUP(ss[-1]); PUSH(v); v = DUP(ss[-1]); PUSH(v));
    CODE("2drop",   DU s = ss.pop(); DROP(s); DROP(tos); tos = ss.pop());
    CODE("2over",   DU v = DUP(ss[-3]); PUSH(v); v = DUP(ss[-3]); PUSH(v));
    CODE("2swap",   DU n = ss.pop(); DU m = ss.pop(); DU l = ss.pop();
                    ss.push(n); PUSH(l); PUSH(m));
    /// @}
    ///@defgroup FPU ops
    ///@{
    CODE("+",       xop2(ADD));
    CODE("-",       xop2(SUB));
    CODE("*",       xop2(MUL));
    CODE("/",       xop2(DIV));
    CODE("mod",     tos = D2I(MOD(ss.pop(), tos)); SCALAR(tos));  /// ( a b -- c )
    CODE("fmod",    tos = MOD(ss.pop(), tos); SCALAR(tos));       /// ( a b -- c ) fmod = x - int(q)*y
    CODE("/mod",                                                  /// ( a b -- c d ) c=a%b, d=a/b
         DU n = ss.pop();
         DU m = MOD(n, tos); SCALAR(m); ss.push(m);
         tos = DIV(n, tos); SCALAR(tos));
    ///@}
    ///@defgroup FPU double precision ops
    ///@{
    CODE("*/",                                                    /// ( a b c -- d ) c= a*b / c
         DU2 n = MUL2(ss.pop(), ss.pop());
         tos = n / tos; SCALAR(tos));
    CODE("*/mod",                                                 /// ( a b c -- d e )
         DU2 n = MUL2(ss.pop(), ss.pop());
         DU2 t = tos;
         DU  m = MOD2(n, tos); SCALAR(m); ss.push(m);
         tos = D2I(n / t));
    ///@}
    ///@defgroup Binary logic ops (convert to integer first)
    ///@{
    CODE("and",     tos = I2D(D2I(tos) & D2I(ss.pop())));
    CODE("or",      tos = I2D(D2I(tos) | D2I(ss.pop())));
    CODE("xor",     tos = I2D(D2I(tos) ^ D2I(ss.pop())));
    CODE("abs",     xop1(ABS));
    CODE("negate",  xop1(NEG));
    CODE("invert",  tos = I2D(~D2I(tos)));
    CODE("rshift",  tos = I2D(D2I(ss.pop()) >> D2I(tos)));
    CODE("lshift",  tos = I2D(D2I(ss.pop()) << D2I(tos)));
    CODE("max",     DU n=ss.pop(); tos = (tos>n) ? tos : n);
    CODE("min",     DU n=ss.pop(); tos = (tos<n) ? tos : n);
    CODE("2*",      tos *= DU1*2);
    CODE("2/",      tos /= DU1*2);
    CODE("1+",      tos += DU1);
    CODE("1-",      tos -= DU1);
    /// @}
    /// @defgroup Data conversion ops
    /// @{
    CODE("f>s",     tos = D2I(tos));     /// nearest-even 0.5 => 0, 1.5 => 2, 2.5 => 2
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
    CODE("decimal", _print(RDX, *BASE=10));
    CODE("hex",     _print(RDX, *BASE=16));
    CODE("bl",      PUSH(0x20));
    CODE("cr",      _print(CR));
    CODE(".",       _print(DOT,  POP()));
    CODE("u.",      _print(UDOT, POP()));
    CODE(".r",      IU i = POPi; sys.dotr(i, POP(), *BASE));
    CODE("u.r",     IU i = POPi; sys.dotr(i, POP(), *BASE, true));
    CODE("type",    POP(); sys.pstr((const char*)MEM(POPi)));     /// pass string pointer
    IMMD("key",     if (compile) add_p(KEY); else PUSH(sys.key()));
    CODE("emit",    _print(EMIT, POP()));
    CODE("space",   _print(SPCS, DU1));
    CODE("spaces",  _print(SPCS, POP()));
    /// @}
    /// @defgroup Literal ops
    /// @{
    IMMD("(",       sys.scan(')'));
    IMMD(".(",      sys.pstr(sys.scan(')')));
    IMMD("\\",      sys.scan('\n'));
    IMMD("s\"",     _quote(STR));
    IMMD(".\"",     _quote(DOTQ));
    /// @}
    /// @defgroup Branching ops
    /// @brief - if...then, if...else...then
    /// @{
    IMMD("if",      PUSH(HERE); add_p(ZBRAN));             /// if    ( -- here )
    IMMD("else",    IU h=HERE;  add_p(BRAN);               /// else ( here -- there )
                    SETJMP(POPi); PUSH(h));
    IMMD("then",    SETJMP(POPi));                         /// backfill jump address
    /// @}
    /// @defgroup Loops
    /// @brief  - begin...again, begin...f until, begin...f while...repeat
    /// @{
    IMMD("begin",   PUSH(HERE));
    IMMD("again",   add_p(BRAN, POPi));                    /// again    ( there -- )
    IMMD("until",   add_p(ZBRAN, POPi));                   /// until    ( there -- )
    IMMD("while",   PUSH(HERE); add_p(ZBRAN));             /// while    ( there -- there here )
    IMMD("repeat",                                         /// repeat    ( there1 there2 -- )
         IU t=POPi; add_p(BRAN, POPi); SETJMP(t));         /// set forward and loop back address
    /// @}
    /// @defgrouop FOR...NEXT loops
    /// @brief  - for...next, for...aft...then...next
    ///    3 for ." f" aft ." a" then i . next  ==> f3 a2 a1 a0 i.e. f once only
    /// @{
    IMMD("for" ,    add_p(FOR); PUSH(HERE));               /// for ( -- here )
    IMMD("next",    add_p(NEXT, POPi));                    /// next ( here -- )
    IMMD("aft",                                            /// aft ( here -- here there )
         POP(); IU h=HERE; add_p(BRAN); PUSH(HERE); PUSH(h));
    /// @}
    /// @}
    /// @defgrouop DO..LOOP loops
    /// @{
    IMMD("do" ,     add_p(DO); PUSH(HERE));                /// do ( -- here )
    CODE("i",       PUSH(rs[-1]));
    CODE("leave",   rs.pop(); rs.pop(); UNNEST());         /// quit DO..LOOP
    IMMD("loop",    add_p(LOOP, POPi));                    /// next ( here -- )
    /// @}
    /// @defgrouop return stack ops
    /// @{
    CODE(">r",      rs.push(POP()));
    CODE("r>",      PUSH(rs.pop()));
    CODE("r@",      PUSH(DUP(rs[-1])));                    /// same as I (the loop counter)
    /// @}
    /// @defgrouop Compiler ops
    /// @{
    CODE("[",       compile = false);
    CODE("]",       compile = true);
    CODE(":",       compile = _word());
    IMMD(";",       add_p(EXIT); compile = false);
    CODE("variable",                                        /// create a variable
         if (!_word()) return;
         add_p(VAR, 0, true); add_du(DU0));                 /// default DU0
    CODE("constant",                                        /// create a constant
         if (!_word()) return;
         add_lit(POP(), true));
    CODE("value",   
         if (!_word()) return;
         add_p(LIT, 0, true, true);                         /// forced extended, TO can update
         add_du(POP()));             
    IMMD("immediate", dict[-1].imm = true);
    CODE("exit",    UNNEST());                              /// early exit the colon word
    /// @}
    /// @defgroup metacompiler
    /// @brief - dict is directly used, instead of shield by macros
    /// @{
    CODE("exec",   IU w = POP(); call(w));                  /// execute word
    CODE("create",
         if (!_word()) return;
         add_p(VAR, 0, true));
    CODE("does>",
         IU pfa = mmu.last()->pfa;
         while (((Param*)MEM(pfa))->op != VAR && (pfa < (IU)HERE)) {  /// find that VAR
             pfa += sizeof(IU);
         }
         SETJMP(pfa);                                       /// set jmp target
         add_p(BRAN, ip); UNNEST());                        /// jmp to next ip
    IMMD("to", _to_value());                                /// alter the value of a constant, i.e. 3 to x
    IMMD("is", _is_alias());                                /// alias a word, i.e. ' y is x
    CODE("[to]",            /// : xx 3 [to] y ;             /// alter constant in compile mode
         IU a = ((Param*)MEM(ip))->ioff + sizeof(IU);       /// address of constant
         DU d = POP();                                      /// constant value to be
         ip += sizeof(IU);                                  /// skip to next instruction
         if (a < T4_PMEM_SZ) CELL(a) = d;                   /// store tos into constant pfa
         else { ERROR("is %x", a); state = STOP; });
    ///
    /// be careful with memory access, because
    /// it could make access misaligned which cause exception
    ///
    CODE("@",     IU i = POPi; PUSH((DU)CELL(i)));          /// i -- n
    CODE("!",     IU i = POPi; CELL(i) = POP());            /// n i --
    CODE("c@",    IU i = POPi; PUSH(I2D(BYTE(i))));         /// i -- c
    CODE("c!",    IU i = POPi; BYTE(i) = (U8)POPi);         /// c i --
    CODE("+!",    IU i = POPi;                              /// n i --
         DU v = CELL(i) + POP();
         SCALAR(v); CELL(i) = v);
    CODE("?",     IU i = POPi; _print(DOT, CELL(i)));       /// i --
    CODE(",",     DU n = POP();  add_du(n));                /// n -- , compile a cell
    CODE("cells", IU i = POPi; PUSH(i * sizeof(DU)));       /// n -- n'
    CODE("allot",                                           /// n --
         IU n = POPi;                                       /// number of bytes
         for (IU i = 0; i < n; i+=sizeof(DU)) add_du(DU0)); /// zero padding
    CODE("th",    IU i = POPi; tos += i * sizeof(DU));      /// w i -- w'
    /// @}
#if DO_MULTITASK    
    /// @defgroup Multitasking ops
    /// @}
    CODE("task",                                            /// w -- task_id
         IU i = POPi; Code &c = dict[i];                    ///< dictionary index
         if (c.udf) PUSH(task_create(c.pfa));               /// create a task starting on pfa
         else pstr("  ?colon word only\n"));
    CODE("rank",  PUSH(id));                                /// ( -- task_id ) used insided a task
    CODE("start", task_start(POPi));                        /// ( task_id -- )
    CODE("join",  join(POPi));                              /// ( task_id -- )
    CODE("lock",  io_lock());                               /// wait for IO semaphore
    CODE("unlock",io_unlock());                             /// release IO semaphore
    CODE("send",  IU t = POPi; send(t, POPi));              /// ( v1 v2 .. vn n tid -- ) pass values onto task's stack
    CODE("recv",  recv());                                  /// ( -- v1 v2 .. vn ) waiting for values passed by sender
    CODE("bcast", bcast(POPi));                             /// ( v1 v2 .. vn -- )
    CODE("pull",  IU t = POPi; pull(t, POPi));              /// ( n task_id -- v1 v2 .. vn )
    /// @}
#endif // DO_MULTITASK    
    /// @defgroup Debug ops
    /// @{
    CODE("abort", tos = -DU1; ss.clear(); rs.clear());      /// clear ss, rs
    CODE("here",  PUSH(HERE));
    CODE("'",     IU w = FIND(sys.fetch()); if (w) PUSH(w));
    CODE(".s",    _ss_dump());
    CODE("depth", PUSH(ss.idx - 1));
    CODE("words", sys.op(OP_WORDS));
    CODE("dict",  sys.op(OP_DICT));                         /// dict_dump in host mode
    CODE("dict_dump", mmu.dict_dump());
    CODE("see",   IU w = FIND(sys.fetch()); if (!w) return;
                  sys.op(OP_SEE, DU0, *BASE, w));
    CODE("dump",  IU n = POPi; DU a = POP();
                  sys.op(OP_DUMP, a, 0, n));
    CODE("forget", _forget());
    CODE("trace", sys.trace(POPi));                         /// set debug/trace level
    /// @}
    /// @defgroup OS ops
    /// @{
    CODE("mstat", mmu.status(true));
//    CODE("rnd",   PUSH(sys.rand(DU1, NORMAL)));             /// generate random number
    CODE("ms",    System::delay(POPi));
    CODE("flush", syscall(OP_FLUSH));                       /// flush output stream
//    CODE("included",                                      /// include external file
//         POP();                                           /// string length, not used
//         sys.load(MEM(POP())));                           /// include external file
    CODE("clock", DU t = System::clock(); SCALAR(t); PUSH(t));
    CODE("bye",   state = STOP);                            /// atomicExch(&state, STOP)
    ///@}
    CODE("boot",  mmu.clear(FIND((char*)"boot") + 1));
#if 0  /* words TODO */
    CODE("power",  {}); /// power(2, 3)
    CODE("?do",    {});
    CODE("roll",   {});
    CODE("u<",     {});
    CODE("u>",     {});
    CODE("within", {});
#endif
    TRACE("ForthVM[%d]::init ok, dict=%p, sizeof(Code)=%ld, sizoef(Param)=%ld\n",
          id, dict, sizeof(Code), sizeof(Param));
}
///======================================================================
///
/// parse input idiom as a word
///
__HOST__ IU
ForthVM::parse(char *idiom) {
    IU w = FIND(idiom);                   /// * search through dictionary
    if (!w) {                             /// * input word not found
        DEBUG(" '%s' not found\n", idiom);
        return 0;                         /// * next, try as a number
    }
    Code &c = dict[w];
    DEBUG("%06x[%3x]%c%c %s ",
         c.udf ? c.pfa : mmu.XTOFF(c.xt), w,
         c.imm ? '*' : ' ', c.udf ? 'u' : ' ',
         c.name);
    
    if (compile && !c.imm) {              /// * in compile mode?
        add_w(w);                         /// * add found word to new colon word
    }
    else { ip = DU0; call(w); }           /// * execute forth word

    return w;
}
///
/// parse input idiom as a number
///
__HOST__ DU
ForthVM::number(char *idiom, char **p) {
    int b = *BASE;
    switch (*idiom) {                     ///> base override
    case '%': b = 2;  idiom++; break;
    case '&':
    case '#': b = 10; idiom++; break;
    case '$': b = 16; idiom++; break;
    }
    DU2 d2 = (b==10 && STRCHR(idiom, '.'))
        ? STRTOF(idiom, p)
        : STRTOL(idiom, p, b);
    if (**p != '\0') {                    /// * not a number, bail
        ERROR(" number(%s) base=%d => error\n", idiom, b);
        return DU0;
    }
    /// is a number
    DU n = (DU)d2;
    char *x = (char*)&n;
    for (int i=0; i<sizeof(DU); i++, x++) {
        const char h2c[] = "0123456789abcdef";
        DEBUG("%c%c ", h2c[((*x)>>4)&0xf], h2c[(*x)&0xf]);
    }
    return n;
}
///
///@name misc eForth functions (in Standard::Core section)
///@{
__HOST__ int
ForthVM::_word() {                        ///< check input word
    char *name = sys.fetch();
    if (name[0]=='\0') {                  /// * missing name?
        sys.pstr(" name?", CR); return 0;
    }
    IU w = FIND(name);
    DEBUG("_find(%s) => %d\n", name, w);
    if (w) {                              /// * word redefined?
        sys.pstr(name);
        sys.pstr(" reDef? ", CR);
    }
    mmu.colon(name);                      /// * create a colon word
    return 1;                             /// * created OK
}
__HOST__ void
ForthVM::_forget() {
    IU w = FIND(sys.fetch()); if (!w) return; /// bail, if not found
    IU b = FIND((char*)"boot")+1;
    mmu.clear(w > b ? w : b);
}
__HOST__ void
ForthVM::_quote(prim_op op) {
    const char *s = sys.scan('"')+1;      ///> string skip first blank
    if (compile) {
        add_p(op, ALIGN(STRLEN(s)+1));    ///> dostr, (+parameter field)
        add_str(s);                       ///> byte0, byte1, byte2, ..., byteN
    }
    else {
        IU h0  = HERE;                    ///> keep current memory addr
        DU len = add_str(s);              ///> write string to PAD
        switch (op) {
        case STR:  PUSH(h0); PUSH(len);            break; ///> addr, len
        case DOTQ: sys.pstr((const char*)MEM(h0)); break; ///> to console
        default:   sys.pstr("_quote unknown op:");
        }
        mmu.set_here(h0);                 ///> restore memory addr
    }
}
__HOST__ void
ForthVM::_to_value() {                    ///> update a constant/value
    IU w = state==QUERY ? FIND(sys.fetch()) : POPi;       ///< constant addr
    if (!w) return;
    if (compile) {
        add_lit((DU)w);                                   /// save addr on stack
        add_w(FIND((char*)"to"));                         /// encode to opcode
    }
    else {
        U8    *pfa = MEM(dict[w].pfa);                    /// fetch constant pointer
        Param &p   = *(Param*)(pfa);
        if (p.op==LIT) {
            *(DU*)(pfa + sizeof(IU)) = POP();             /// update constant value
        }
    }
}
__HOST__ void
ForthVM::_is_alias() {                                    /// create alias function
    IU w = state==QUERY ? FIND(sys.fetch()) : POPi;       ///< word addr
    if (!w) return;
    if (compile) {
        add_lit((DU)w);                                   /// save addr on stack
        add_w(FIND((char*)"is"));
    }
    else dict[POPi].xt = dict[w].xt;
}

__HOST__ void
ForthVM::_ss_dump() {
    sys.dots(id, tos, ss.idx, *BASE);
}

__HOST__ void
ForthVM::_print(io_op o, DU v) {
    sys.dot(o, v);
#if T4_DO_OBJ
    if (IS_OBJ(v)) {
        mmu.mark_free(v);                ///< release memory
        state = HOLD;                    ///< forced flush (memory safe)
    }
#endif // T4_DO_OBJ
}
        
#if (T4_DO_OBJ && T4_DO_NN)
__HOST__ int
ForthVM::_ds_next(U32 ioff) {
    T4Base &m = mmu.du2obj(tos);        ///< Network Model
    if (!m.is_model()) {
        ERROR("TOS is not a network model?\n"); return 0;
    }
    T4Base &d = mmu.du2obj(rs[-1]);     ///< RTOS is fetch dataset
    if (!d.is_dataset()) {
        ERROR("RTOS is not a dataset?\n"); return 0;
    }
    if (((Dataset&)d).done) {           /// * Dataset one epoch completed
        DU v = rs.pop();                /// * pop off dataset
        DROP(v);                        /// * free memory if a physical dataset
        ((Model&)m).tick();             /// * bump epoch counter
    }
    else {
        syscall(OP_FETCH, rs[-1], 0);   /// * issue a dataset fetch
        ip = ioff;                      /// * loop branch target address
    }
    return 1;
}
#endif // (T4_DO_OBJ && T4_DO_NN)
///@}

} // namespace t4::vm
//=======================================================================================
