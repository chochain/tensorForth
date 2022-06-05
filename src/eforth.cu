/**
 * @file
 * @brief - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "mmu.h"
#include "eforth.h"
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
///@name Data conversion
///@{
#define INT(f)    (static_cast<int>(f + 0.5))   /**< cast float to int                       */
#define I2D(i)    (static_cast<DU>(i))          /**< cast int back to float                  */
#define ABS(d)    (fabs(d))                     /**< absolute value                          */
#define ZERO(d)   (ABS(d) < DU_EPS)             /**< zero check                              */
#define MOD(t,n)  (fmod(t, n))                  /**< fmod two floating points                */
#define BOOL(f)   ((f) ? -1 : 0)                /**< default boolean representation          */
///@}
///@name Tensor ops
///@{
#define TOPx      (*ptop &= ~1)                 /**< tensor flag mask for top                */
#define RANK      ((*ptop & 6) >> 1)            /**< tensor rank of top                      */
///@}
///@name Dictioanry access
///@{
#define PFA(w)    (dict[(IU)(w)].pfa)           /**< PFA of given word id                    */
#define HERE      (mmu.here())                  /**< current context                         */
#define SETJMP(a) (mmu.setjmp(a))               /**< address offset for branching opcodes    */
#define FIND(s)   (mmu.find(s, compile, ucase)) /**< find input idiom in dictionary          */
#define EXEC(w)   ((*(dict[(IU)(w)].xt))())     /**< execute primitive word                  */
///@}
///@name Heap memory load/store macros
///@{
#define POPi      (INT(POP()))                  /**< convert popped DU as an IU              */
#define LDi(ip)   (mmu.ri((IU)(ip)))            /**< read an instruction unit from pmem      */
#define LDd(ip)   (mmu.rd((IU)(ip)))            /**< read a data unit from pmem              */
#define STd(ip,d) (mmu.wd((IU)(ip), (DU)(d)))   /**< write a data unit to pmem               */
#define LDs(ip)   (mmu.mem((IU)(ip)))           /**< pointer to IP address fetched from pmem */
///@}
__GPU__
ForthVM::ForthVM(int khz, Istream *istr, Ostream *ostr, MMU *mmu0)
    : khz(khz), fin(*istr), fout(*ostr), mmu(*mmu0), dict(mmu0->dict()) {
        T4_TRACE("FVM dict=%p, mem=%p, vss=%p\n", dict, mmu.mem(0), mmu.vss(blockIdx.x));
}
///
/// Forth inner interpreter (colon word handler)
///
__GPU__ char*
ForthVM::next_idiom()  {                             ///< get next idiom from input stream
    fin >> idiom; return idiom;
}
__GPU__ char*
ForthVM::scan(char delim) {                          ///< scan input stream for delimiter
    fin.get_idiom(idiom, delim); return idiom;
}
__GPU__ void
ForthVM::nest() {
    int dp = 0;                                      ///< iterator depth control
    while (dp >= 0) {
        IU w = LDi(IP);                              ///< fetch opcode, and cache dataline hopefully
        while (w != EXIT) {                          ///< loop till EXIT
            IP += sizeof(IU);                        ///< ready IP for next opcode
            if (dict[w].def) {                       ///< is it a colon word?
                rs.push(WP);                         ///< * setup callframe (ENTER)
                rs.push(IP);
                IP = PFA(w);                         ///< jump to pfa of given colon word
                dp++;                                ///< go one level deeper
            }
            else if (w == DONEXT) {                  ///< DONEXT handler (save 600ms / 100M cycles on Intel)
                if ((rs[-1] -= 1) >= 0) IP = LDi(IP);///< decrement loop counter, and fetch target addr
                else { IP += sizeof(IU); rs.pop(); } ///< done loop, pop off loop counter
            }
            else EXEC(w);                            ///< execute primitive word
            w = LDi(IP);                             ///< fetch next opcode
        }
        if (dp-- > 0) {                              ///< pop off a level
            IP = INT(rs.pop());                      ///< * restore call frame (EXIT)
            WP = INT(rs.pop());
        }
        yield();                                     ///< give other tasks some time
    }
}
///
/// Dictionary compiler proxy macros to reduce verbosity
///
__GPU__ __INLINE__ void ForthVM::add_w(IU w)  {
    add_iu(w);
    T4_TRACE("add_w(%d) => %s\n", w, dict[w].name);
}
__GPU__ __INLINE__ void ForthVM::add_iu(IU i) { mmu.add((U8*)&i, sizeof(IU)); }
__GPU__ __INLINE__ void ForthVM::add_du(DU d) { mmu.add((U8*)&d, sizeof(DU)); }
__GPU__ __INLINE__ void ForthVM::add_str(const char *s) {
    int sz = STRLENB(s)+1; sz = ALIGN2(sz);          ///> calculate string length, then adjust alignment (combine?)
    mmu.add((U8*)s, sz);
}
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    if (dict[w].def) { WP = w; IP = dict[w].pfa; nest(); }
    else (*(FPTR)((UFP)dict[w].xt & ~0x3))();        ///> execute function pointer (strip off immdiate bit)
}
///==============================================================================
///
/// debug functions
///
__GPU__ __INLINE__ void ForthVM::dot(DU v)          { fout << ' ' << v; }
__GPU__ __INLINE__ void ForthVM::dot_r(int n, DU v) { fout << setw(n) << v; }
__GPU__ __INLINE__ void ForthVM::ss_dump(int n) {
    ss[T4_SS_SZ-1] = top;        // put top at the tail of ss (for host display)
    fout << opx(OP_SS, n);
}
///
/// global memory access macros
///
#define PEEK(a)        (U8)(*(U8*)((UFP)(a)))
#define POKE(a, c)     (*(U8*)((UFP)(a))=(U8)(c))
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
    CODE("exit",    WP = INT(rs.pop()); IP = INT(rs.pop())),        // quit current word execution
    CODE("donext",
         if ((rs[-1] -= 1) >= 0) IP = LDi(IP);
         else { IP += sizeof(IU); rs.pop(); }),
    CODE("dovar",   PUSH(IP); IP += sizeof(DU)),
    CODE("dolit",   PUSH(LDd(IP)); IP += sizeof(DU)),
    CODE("dostr",
        char *s  = (char*)LDs(IP);                        // get string pointer
        int  sz  = STRLENB(s)+1;
        PUSH(IP); IP += ALIGN2(sz)),
    CODE("dotstr",
        char *s  = (char*)LDs(IP);                        // get string pointer
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
    CODE("r@",   PUSH(rs[-1])),
    ///@}
    ///@defgroup Stack ops
    ///@brief - opcode sequence can be changed below this line
    ///@{
    CODE("dup",  PUSH(top)),
    CODE("drop", top = ss.pop()),
    CODE("over", PUSH(ss[-1])),
    CODE("swap", DU n = ss.pop(); PUSH(n)),
    CODE("rot",  DU n = ss.pop(); DU m = ss.pop(); ss.push(n); PUSH(m)),
    CODE("pick", int i = INT(top); top = ss[-i]),
    ///@}
    ///@defgroup Stack double
    ///@{
    CODE("2dup", PUSH(ss[-1]); PUSH(ss[-1])),
    CODE("2drop",ss.pop(); top = ss.pop()),
    CODE("2over",PUSH(ss[-3]); PUSH(ss[-3])),
    CODE("2swap",
        DU n = ss.pop(); DU m = ss.pop(); DU l = ss.pop();
        ss.push(n); PUSH(l); PUSH(m)),
    ///@}
    ///@defgroup FPU ops
    ///@{
    CODE("+",    top += ss.pop()),                    /// TODO: vector.add
    CODE("*",    top *= ss.pop()),                    /// TODO: matmul
    CODE("-",    top =  ss.pop() - top),
    CODE("/",    top =  ss.pop() / top; TOPx),
    CODE("mod",  top =  MOD(ss.pop(), top)),          /// fmod = x - int(q)*y
    CODE("/mod",
        DU n = ss.pop(); DU t = top;
        ss.push(MOD(n, t)); top = n / t),
    ///@}
    ///@defgroup FPU double precision ops
    ///@{
    CODE("*/",   top =  (DU2)ss.pop() * ss.pop() / top),
    CODE("*/mod",
        DU2 n = (DU2)ss.pop() * ss.pop();
        ss.push(MOD(n, top)); top = round(n / top)),
    ///@}
    ///@defgroup Binary logic ops (convert to integer first)
    ///@{
    CODE("and",  top = I2D(INT(ss.pop()) & INT(top))),
    CODE("or",   top = I2D(INT(ss.pop()) | INT(top))),
    CODE("xor",  top = I2D(INT(ss.pop()) ^ INT(top))),
    CODE("abs",  top = ABS(top)),
    CODE("negate", top *= -1),
    CODE("max",  DU n=ss.pop(); top = (top>n)?top:n),
    CODE("min",  DU n=ss.pop(); top = (top<n)?top:n),
    CODE("2*",   top *= 2),
    CODE("2/",   top /= 2),
    CODE("1+",   top += 1),
    CODE("1-",   top -= 1),
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
    CODE("0<",   top = BOOL(top <  DU_EPS)),
    CODE("0>",   top = BOOL(top >  -DU_EPS)),
    CODE("=",    top = BOOL(ZERO(ss.pop() - top))),
    CODE(">",    top = BOOL((ss.pop() -  top) > DU_EPS)),
    CODE("<",    top = BOOL((ss.pop() -  top) < -DU_EPS)),
    CODE("<>",   top = BOOL(!ZERO(ss.pop() - top))),
    CODE(">=",   top = BOOL((ss.pop() - top) >= DU_EPS)),      // pretty much the same as > for float
    CODE("<=",   top = BOOL((ss.pop() - top) <= -DU_EPS)),
    ///@}
    ///@defgroup IO ops
    ///@{
    CODE("base@",   PUSH(I2D(radix))),
    CODE("base!",   fout << setbase(radix = INT(POP()))),
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
    ///@}
    ///@defgroup Literal ops
    ///@{
    CODE("[",       compile = false),
    CODE("]",       compile = true),
    IMMD("(",       scan(')')),
    IMMD(".(",      fout << scan(')')),
    CODE("\\",      scan('\n')),
    CODE("$\"",
        const char *s = scan('"')+1;        // string skip first blank
        add_w(DOSTR);
        add_str(s)),                        // dostr, (+parameter field)
    IMMD(".\"",
        const char *s = scan('"')+1;        // string skip first blank
        add_w(DOTSTR);
        add_str(s)),                        // dotstr, (+parameter field)
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
    IMMD(";", add_w(EXIT); compile = false),
    CODE("variable",                                        // create a variable
        mmu.colon(next_idiom());                            // create a new word on dictionary
        add_w(DOVAR);                                       // dovar (+parameter field)
        add_du(0);                                          // data storage (32-bit integer now)
        add_w(EXIT)),
    CODE("constant",                                        // create a constant
        mmu.colon(next_idiom());                            // create a new word on dictionary
        add_w(DOLIT);                                       // dovar (+parameter field)
        add_du(POP());                                      // data storage (32-bit integer now)
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
        STd(PFA(w) + sizeof(IU), POP())),
    CODE("is",              // ' y is x                     // alias a word
        int w = FIND(next_idiom());                         // can serve as a function pointer
        mmu.wi(PFA(POPi), PFA(w))),                         // but might leave a dangled block
    CODE("[to]",            // : xx 3 [to] y ;              // alter constant in compile mode
        IU w = LDi(IP); IP += sizeof(IU);                   // fetch constant pfa from 'here'
        STd(PFA(w) + sizeof(IU), POP())),
    ///
    /// be careful with memory access, because
    /// it could make access misaligned which cause exception
    ///
    CODE("@",     IU w = POPi; PUSH(LDd(w))),                                     // w -- n
    CODE("!",     IU w = POPi; STd(w, POP())),                                    // n w --
    CODE(",",     DU n = POP(); add_du(n)),
    CODE("allot", DU v = 0; for (IU n = POPi, i = 0; i < n; i++) add_du(v)),      // n --
    CODE("+!",    IU w = POPi; STd(w, LDd(w)+POP())),                             // n w --
    CODE("?",     IU w = POPi; fout << LDd(w) << " "),                            // w --
    ///@}
    ///@defgroup Debug ops
    ///@{
    CODE("here",  PUSH(HERE)),
    CODE("ucase", ucase = !ZERO(POPi)),
    CODE("'",     int w = FIND(next_idiom()); PUSH(w)),
    CODE("pfa",   int w = FIND(next_idiom()); PUSH(PFA(w))),
    CODE(".s",    ss_dump(POPi)),
    CODE("words", fout << opx(OP_WORDS)),
    CODE("see",   int w = FIND(next_idiom()); fout << opx(OP_SEE, w)),
    CODE("dump",  int n = POPi; int a = POPi; fout << opx(OP_DUMP, a, n)),
    CODE("forget",
        int w = FIND(next_idiom());
        if (w<0) return;
        IU b = FIND("boot")+1;
        mmu.clear(w > b ? w : b)),
    ///@}
    ///@defgroup System ops
    ///@{
    CODE("peek",  IU a = POPi; PUSH(PEEK(a))),
    CODE("poke",  IU a = POPi; POKE(a, POPi)),
    CODE("clock", PUSH(clock()/khz)),
    CODE("delay", delay(POPi)),                                // TODO: change to VM_WAIT
    ///@}
    ///@defgroup Tensor ops
    ///@brief - adhere to PyTorch naming
    ///@{
    CODE("T[",      {}),                 ///< TODO: 1-D tensor
    CODE("T2[",     {}),                 ///< TODO: 2-D tensor
    CODE("T3[",     {}),                 ///< TODO: 3-D tensor
    CODE("zeros",   {}),                 ///< TODO: zeros
    CODE("ones",    {}),                 ///< TODO: ones
    CODE("rand",    {}),                 ///< TODO: rand
    CODE("randn",   {}),                 ///< TODO: randn
    CODE("matmul",  {}),                 ///< TODO: matmul
    CODE("view2",   {}),                 ///< TODO: 
    CODE("view3",   {}),                 ///< TODO: 
    ///@}
    CODE("bye",   status = VM_STOP),
    CODE("boot",  mmu.clear(FIND("boot") + 1))
    };
    int n  = sizeof(prim)/sizeof(Code); ///< number of primitive entries
    UFP x0 = ~0;                        ///< base of xt   allocations
    UFP n0 = ~0;                        ///< base of name allocations
    for (int i=0; i<n; i++) {
        mmu << (Code*)&prim[i];         ///> TODO: prevent deep copy
        if ((UFP)dict[i].xt   < x0) x0 = (UFP)dict[i].xt;
        if ((UFP)dict[i].name < n0) n0 = (UFP)dict[i].name;
    }
    for (int i=0; i<n; i++) {           ///< dump dictionary from device
        MMU_TRACE("%3d> xt=%4x:%p name=%4x:%p %s\n", i,
            (U16)((UFP)dict[i].xt   - x0), dict[i].xt,
            (U16)((UFP)dict[i].name - n0), dict[i].name,
            dict[i].name);
    }
    MMU_TRACE("VM=%p dict[%d] sizeof(Code)=%d sizeof(Tensor)=%d\n",
        this, n, (int)sizeof(Code), (int)sizeof(Tensor));

    status = VM_RUN;
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
__GPU__ void
ForthVM::outer() {
    while (fin >> idiom) {                   /// loop throught tib
        T4_TRACE("%d>> %-10s => ", blockIdx.x, idiom);
        int w = FIND(idiom);                 /// * search through dictionary
        if (w>=0) {                          /// * word found?
            T4_TRACE("%4x:%p %s %d ",
                dict[w].def ? dict[w].pfa : 0,
                dict[w].xt, dict[w].name, w);
            if (compile && !dict[w].immd) {  /// * in compile mode?
                add_w((IU)w);                /// * add found word to new colon word
            }
            else {
                T4_TRACE("=> call(%s)\n", dict[w].name);
                call((IU)w);                 /// * execute forth word
            }
            continue;
        }
        // try as a number
        char *p;
        DU n = (STRCHR(idiom, '.'))
                ? STRTOF(idiom, &p)
                : STRTOL(idiom, &p, radix);
        if (*p != '\0') {                    /// * not number
            fout << idiom << "? " << ENDL;   ///> display error prompt
            compile = false;                 ///> reset to interpreter mode
            break;                           ///> skip the entire input buffer
        }
        // is a number
        T4_TRACE("%f\n", n);
        if (compile) {                       /// * add literal when in compile mode
            add_w(DOLIT);                    ///> dovar (+parameter field)
            add_du(n);                       ///> store literal
        }
        else {
            T4_TRACE("ss.push(%08x)\n", *(U32*)&n);
            PUSH(n);                         ///> or, add value onto data stack
        }
    }
    if (!compile) ss_dump(ss.idx);
}
//=======================================================================================
