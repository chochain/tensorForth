/*! @file
  @brief
  tensorForth Forth Vritual Machine implementation
*/
#include "mmu.h"
#include "eforth.h"
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
#define INT(f)    (static_cast<int>(f))         /** cast float to int                        */
#define I2D(i)    (static_cast<DU>(i))          /** cast int back to float                   */
#define ABS(d)    (fabs(d))                     /** absolute value                           */
#define ZERO(d)   (ABS(d) < DU_EPS)             /** zero check                               */
#define BOOL(f)   ((f) ? -1 : 0)                /** default boolean representation           */

#define PFA(w)    (dict[(IU)(w)].pfa)           /** PFA of given word id                     */
#define HERE      (mmu.here())                  /** current context                          */
#define XOFF(xp)  (mmu.xtoff((UFP)(xp)))        /** XT offset (index) in code space          */
#define XT(ix)    (mmu.xt(ix))                  /** convert XT offset to function pointer    */
#define SETJMP(a) (mmu.setjmp(a))               /** address offset for branching opcodes     */

#define POPi      (INT(POP()))                  /** convert popped DU as an IU               */
#define FIND(s)   (mmu.find(s, compile, ucase)) /** find input idiom in dictionary           */
///
/// heap memory load/store macros
///
#define LDi(ip)   (mmu.ri((IU)(ip)))            /** read an instruction unit from pmem       */
#define LDd(ip)   (mmu.rd((IU)(ip)))            /** read a data unit from pmem               */
#define STd(ip,d) (mmu.wd((IU)(ip), (DU)(d)))   /** write a data unit to pmem                */
#define LDs(ip)   (mmu.mem((IU)(ip)))           /** pointer to IP address fetched from pmem  */

__GPU__
ForthVM::ForthVM(Istream *istr, Ostream *ostr, MMU *mmu0)
    : fin(*istr), fout(*ostr), mmu(*mmu0), dict(mmu0->dict()) {
#if CC_DEBUG
        printf("D: dict=%p, mem=%p, vss=%p\n", dict, mmu.mem(0), mmu.vss(blockIdx.x));
#endif // CC_DEBUG
}
///
/// Forth inner interpreter (colon word handler)
///
__GPU__ char*
ForthVM::next_idiom()  {                            /// get next idiom from input stream
    fin >> idiom; return idiom;
}
__GPU__ char*
ForthVM::scan(char delim) {                         /// scan input stream for delimiter
    fin.get_idiom(idiom, delim); return idiom;
}
__GPU__ void
ForthVM::nest() {
    int dp = 0;                                      /// iterator depth control
    while (dp >= 0) {
        IU ix = LDi(IP);                             /// fetch opcode
        while (ix) {                                 /// fetch till EXIT
            IP += sizeof(IU);
            if (ix & 1) {
                rs.push(WP);                         /// * setup callframe (ENTER)
                rs.push(IP);
                IP = ix & ~0x1;                      /// word pfa (def masked)
                dp++;                                /// go one level deeper
            }
            else if (ix == NXT) {                    /// DONEXT handler (save 600ms / 100M cycles on Intel)
                if ((rs[-1] -= 1) >= 0) IP = LDi(IP);
                else { IP += sizeof(IU); rs.pop(); }
            }
            else (*(FPTR)XT(ix))(ix);                /// * execute primitive word
            ix = LDi(IP);                           /// * fetch next opcode
        }
        if (dp-- > 0) {                              /// pop off a level
            IP = rs.pop();                           /// * restore call frame (EXIT)
            WP = rs.pop();
        }
        yield();                                     ///> give other tasks some time
    }
}
///
/// Dictionary compiler proxy macros to reduce verbosity
///
__GPU__ __INLINE__ void ForthVM::add_iu(IU i) { mmu.add((U8*)&i, sizeof(IU)); }
__GPU__ __INLINE__ void ForthVM::add_du(DU d) { mmu.add((U8*)&d, sizeof(DU)); }
__GPU__ __INLINE__ void ForthVM::add_str(const char *s) {
    int sz = STRLENB(s)+1; sz = ALIGN2(sz);          ///> calculate string length, then adjust alignment (combine?)
    mmu.add((U8*)s, sz);
}
__GPU__ __INLINE__ void ForthVM::add_w(IU w) {
    Code &c = dict[w];
    IU   ip = c.def ? (c.pfa | 1) : (w==EXIT ? 0 : XOFF(c.xt));
    add_iu(ip);
#if CC_DEBUG
    printf("add_w(%d) => %4x:%p %s\n", w, ip, c.xt, c.name);
#endif // CC_DEBUG
}
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    Code &c = dict[w];
    if (c.def) { WP = w; IP = c.pfa; nest(); }
    else (*(FPTR)(((UFP)c.xt) & ~0x3))(w);
}
///==============================================================================
///
/// debug functions
///
__GPU__ __INLINE__ void ForthVM::dot(DU v)          { fout << v << ' '; }
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
    /// @defgroup Execution flow ops
    /// @brief - DO NOT change the sequence here (see forth_opcode enum)
    /// @{
    CODE("exit",    WP = rs.pop(); IP = rs.pop()),         // quit current word execution
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
    /// @defgroup FPU ops
    /// @{
    CODE("+",    top += ss.pop()),
    CODE("*",    top *= ss.pop()),
    CODE("-",    top =  ss.pop() - top),
    CODE("/",    top =  ss.pop() / top),
    CODE("mod",  top =  fmod(ss.pop(), top)),          /// fmod = x - int(q)*y
    CODE("/mod",
        DU n = ss.pop(); DU t = top;
        ss.push(fmod(n, t)); top = n / t),
	/// @}
	/// @defgroup FPU double precision ops
	/// @{
	CODE("*/",   top =  (DU2)ss.pop() * ss.pop() / top),
    CODE("*/mod",
        DU2 n = (DU2)ss.pop() * ss.pop();  DU t = top;
        ss.push(fmod(n, t)); top = round(n / t)),
	/// @}
	/// @defgroup binary logic ops (convert to integer first)
	/// @{
    CODE("and",  top = I2D(INT(ss.pop()) & INT(top))),
    CODE("or",   top = I2D(INT(ss.pop()) | INT(top))),
    CODE("xor",  top = I2D(INT(ss.pop()) ^ INT(top))),
    CODE("abs",  top = ABS(top)),
	CODE("negate", top = -top),
    CODE("max",  DU n=ss.pop(); top = (top>n)?top:n),
    CODE("min",  DU n=ss.pop(); top = (top<n)?top:n),
    CODE("2*",   top *= 2),
    CODE("2/",   top /= 2),
    CODE("1+",   top += 1),
    CODE("1-",   top -= 1),
	/// @}
	/// @defgroup data conversion ops
	/// @{
	CODE("int",  top = INT(top)),                /// integer part, 1.5 => 1, -1.5 => -1
	CODE("round",top = round(top)),              /// rounding 1.5 => 2, -1.5 => -1
	CODE("ceil", top = ceil(top)),
	CODE("floor",top = floor(top)),
    /// @}
    /// @defgroup Logic ops
    /// @{
    CODE("0= ",  top = BOOL(ZERO(top))),
    CODE("0<",   top = BOOL(top <  DU_EPS)),
    CODE("0>",   top = BOOL(top >  -DU_EPS)),
    CODE("=",    top = BOOL(ZERO(ss.pop() - top))),
    CODE(">",    top = BOOL((ss.pop() -  top) > DU_EPS)),
    CODE("<",    top = BOOL((ss.pop() -  top) < -DU_EPS)),
    CODE("<>",   top = BOOL(!ZERO(ss.pop() - top))),
    CODE(">=",   top = BOOL((ss.pop() - top) >= DU_EPS)),      // pretty much the same as > for float
    CODE("<=",   top = BOOL((ss.pop() - top) <= -DU_EPS)),
    /// @}
    /// @defgroup IO ops
    /// @{
    CODE("base@",   PUSH(radix)),
    CODE("base!",   fout << setbase(radix = POP())),
    CODE("hex",     fout << setbase(radix = 16)),
    CODE("decimal", fout << setbase(radix = 10)),
    CODE("cr",      fout << ENDL),
    CODE(".",       dot(POP())),
    CODE(".r",      int n = POPi; dot_r(n, POP())),
    CODE("u.r",     int n = POPi; dot_r(n, ABS(POP()))),
    CODE(".f",      int n = POPi; fout << setprec(n) << POP()),
    CODE("key",     PUSH(next_idiom()[0])),
    CODE("emit",    fout << (char)POP()),
    CODE("space",   fout << ' '),
    CODE("spaces",
         int n = POP();
         MEMSET(idiom, ' ', n); idiom[n] = '\0';
         fout << idiom),
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
        add_w(DOSTR);
        add_str(s)),                        // dostr, (+parameter field)
    IMMD(".\"",
        const char *s = scan('"')+1;        // string skip first blank
        add_w(DOTSTR);
        add_str(s)),                        // dotstr, (+parameter field)
    /// @}
    /// @defgroup Branching ops
    /// @brief - if...then, if...else...then
    /// @{
    IMMD("if", add_w(ZBRAN); PUSH(HERE); add_iu(0)),        // if   ( -- here )
    IMMD("else",                                            // else ( here -- there )
        add_w(BRAN);
        IU h = HERE; add_iu(0); SETJMP(POP()); PUSH(h)),    // set forward jump
    IMMD("then", SETJMP(POP())),                            // backfill jump address
    /// @}
    /// @defgroup Loops
    /// @brief  - begin...again, begin...f until, begin...f while...repeat
    /// @{
    IMMD("begin",   PUSH(HERE)),
    IMMD("again",   add_w(BRAN);  add_iu(POP())),           // again    ( there -- )
    IMMD("until",   add_w(ZBRAN); add_iu(POP())),           // until    ( there -- )
    IMMD("while",   add_w(ZBRAN); PUSH(HERE); add_iu(0)),   // while    ( there -- there here )
    IMMD("repeat",  add_w(BRAN);                            // repeat    ( there1 there2 -- )
        IU t=POPi; add_iu(POPi); SETJMP(t)),                // set forward and loop back address
    /// @}
    /// @defgrouop For loops
    /// @brief  - for...next, for...aft...then...next
    /// @{
    IMMD("for" ,    add_w(TOR); PUSH(HERE)),                // for ( -- here )
    IMMD("next",    add_w(DONEXT); add_iu(POP())),          // next ( here -- )
    IMMD("aft",                                             // aft ( here -- here there )
        POP(); add_w(BRAN);
        IU h=HERE; add_iu(0); PUSH(HERE); PUSH(h)),
    /// @}
    /// @defgrouop Compiler ops
    /// @{
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
    /// @}
    /// @defgroup metacompiler
    /// @brief - dict is directly used, instead of shield by macros
    /// @{
    CODE("exec",  call(POP())),                              // execute word
    CODE("create",
        mmu.colon(next_idiom());                            // create a new word on dictionary
        add_w(DOVAR)),                                      // dovar (+ parameter field)
    CODE("to",              // 3 to x                       // alter the value of a constant
        int w = FIND(next_idiom());                         // to save the extra @ of a variable
        STd(PFA(w) + sizeof(IU), POP())),
    CODE("is",              // ' y is x                     // alias a word
        int w = FIND(next_idiom());                         // can serve as a function pointer
        mmu.wi(PFA(POP()), PFA(w))),                        // but might leave a dangled block
    CODE("[to]",            // : xx 3 [to] y ;              // alter constant in compile mode
        IU w = LDi(IP); IP += sizeof(IU);                // fetch constant pfa from 'here'
        STd(PFA(w) + sizeof(IU), POP())),
    ///
    /// be careful with memory access, especially BYTE because
    /// it could make access misaligned which slows the access speed by 2x
    ///
    CODE("@",     IU w = POPi; PUSH(LDd(w))),                                     // w -- n
    CODE("!",     IU w = POPi; STd(w, POP())),                                    // n w --
    CODE(",",     DU n = POP(); add_du(n)),
    CODE("allot", DU v = 0; for (IU n = POPi, i = 0; i < n; i++) add_du(v)),      // n --
    CODE("+!",    IU w = POPi; STd(w, LDd(w)+POP())),                             // n w --
    CODE("?",     IU w = POPi; fout << LDd(w) << " "),                            // w --
    /// @}
    /// @defgroup Debug ops
    /// @{
    CODE("here",  PUSH(HERE)),
    CODE("ucase", ucase = POP()),
    CODE("'",     int w = FIND(next_idiom()); PUSH(w)),
	CODE("pfa",   int w = FIND(next_idiom()); PUSH(PFA(w))),
    CODE(".s",    ss_dump(POP())),
    CODE("words", fout << opx(OP_WORDS)),
    CODE("see",   int w = FIND(next_idiom()); fout << opx(OP_SEE, w)),
    CODE("dump",  DU n = POP(); DU a = POP(); fout << opx(OP_DUMP, a, n)),
    CODE("forget",
        int w = FIND(next_idiom());
        if (w<0) return;
        IU b = FIND("boot")+1;
        mmu.clear(w > b ? w : b)),
    /// @}
    /// @defgroup System ops
    /// @{
    CODE("peek",  IU a = POP(); PUSH(PEEK(a))),
    CODE("poke",  IU a = POP(); POKE(a, POP())),
    CODE("clock", PUSH(millis())),
    CODE("delay", delay(POP())),                                // TODO: change to VM_WAIT
    CODE("bye",   status = VM_STOP),
    CODE("boot",  mmu.clear(FIND("boot") + 1))
    /// @}
    };
    int n = sizeof(prim)/sizeof(Code);
    for (int i=0; i<n; i++) {
        mmu << (Code*)&prim[i];
    }
    NXT = XOFF(dict[DONEXT].xt);         /// cache offset to subroutine address
#if CC_DEBUG
	for (int i=0; i<n; i++) {
	    printf("%3d> xt=%4x:%p name=%4x:%p %s\n", i,
				XOFF(dict[i].xt), dict[i].fp,
				(dict[i].name - dict[0].name), dict[i].name,
				dict[i].name);           /// dump dictionary from device
	}
#endif // CC_DEBUG
    printf("init() VM=%p sizeof(Code)=%d\n", this, (int)sizeof(Code));
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
#if CC_DEBUG
        printf("%d>> %s => ", blockIdx.x, idiom);
#endif // CC_DEBUG
        int w = FIND(idiom);                 /// * search through dictionary
        if (w>=0) {                          /// * word found?
#if CC_DEBUG
            printf("%4x:%p %s %d\n",
            	dict[w].def ? dict[w].pfa : XOFF(dict[w].xt),
            	dict[w].xt, dict[w].name, w);
#endif // CC_DEBUG
            if (compile && !dict[w].immd) {  /// * in compile mode?
                add_w((IU)w);                /// * add found word to new colon word
            }
            else call((IU)w);                /// * execute forth word
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
#if CC_DEBUG
        printf("%f = %08x\n", n, *(U32*)&n);
#endif // CC_DEBUG
        if (compile) {                       /// * add literal when in compile mode
            add_w(DOLIT);                    ///> dovar (+parameter field)
            add_du(n);                       ///> store literal
        }
        else PUSH(n);                        ///> or, add value onto data stack
    }
    if (!compile) ss_dump(ss.idx);
}
//=======================================================================================
