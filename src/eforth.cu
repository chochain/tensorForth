/*! @file
  @brief
  cueForth Forth Vritual Machine implementation
*/
#include "mmu.h"
#include "eforth.h"
///
/// Forth Virtual Machine operational macros
///
#define INT(f)    (static_cast<int>(f+0.5f))  /** cast float to int                        */
#define I2DU(i)   (static_cast<DU>(i))        /** cast int back to float                   */
#define LWIP      (mmu[-1].plen)              /** parameter field tail of latest word      */
#define JMPIP     (IP0 + *(IU*)IP)            /** branching target address                 */
#define IPOFF     ((IU)(IP - PMEM0))          /** IP offset relative parameter memory root */
#define FIND(s)   (mmu.find(s, compile, ucase))
#define POPi      ((IU)INT(POP()))

__GPU__
ForthVM::ForthVM(Istream *istr, Ostream *ostr, MMU *mmu0)
    : fin(*istr), fout(*ostr), mmu(*mmu0) {
    PMEM0 = IP0 = IP = mmu.mem0();
    printf("D: dict=%p, mem=%p, vss=%p\n", &mmu[0], PMEM0, mmu.vss(blockIdx.x));
}
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
    IP0 = IP = mmu.pfa(WP=c);               // CC: this takes 30ms/1K, need work
//  try                                     // kernal does not support exception
    {                                       // CC: is dict[c] kept in cache?
        U8 *ipx = IP + mmu[c].plen;         // CC: this saves 350ms/1M
        while (IP < ipx) {                  /// * recursively call all children
            IU c1 = *IP; IP += sizeof(IU);  // CC: cost of (ipx, c1) on stack?
            call(c1);                       ///> execute child word
        }                                   ///> can do IP++ if pmem unit is 16-bit
    }
//    catch(...) {}                         ///> protect if any exeception
    yield();                                ///> give other tasks some time
    IP0 = mmu.pfa(WP=rs.pop());             /// * restore call frame
    IP  = PMEM0 + INT(rs.pop());
}
///
/// Dict compiler proxy macros to reduce verbosity
///
__GPU__ __INLINE__ void ForthVM::add_iu(IU i) { mmu.add((U8*)&i, sizeof(IU)); }
__GPU__ __INLINE__ void ForthVM::add_du(DU d) { mmu.add((U8*)&d, sizeof(DU)); }
__GPU__ __INLINE__ void ForthVM::add_str(IU op, const char *s) {
    int sz = STRLENB(s)+1; sz = ALIGN2(sz);
    mmu.add((U8*)&op, sizeof(IU));
    mmu.add((U8*)s, sz);
}
__GPU__ __INLINE__ void ForthVM::call(IU w) {
    if (mmu[w].def) nest(w);
    else (*(fop*)(((uintptr_t)mmu[w].xt)&~0x3))(w);
}
///==============================================================================
///
/// debug functions
///
__GPU__ __INLINE__ void ForthVM::dot_r(int n, DU v) { fout << setw(n) << v; }
__GPU__ __INLINE__ void ForthVM::ss_dump(int n) {
    ss[CUEF_SS_SZ-1] = top;        // put top at the tail of ss (for host display)
    fout << opx(OP_SS, n);
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
    CODE("dolit",   PUSH(mmu.rd((DU*)IP)); IP += sizeof(DU)),
    CODE("dostr",
        char *s  = (char*)IP;                     // get string pointer
        int  sz  = STRLENB(s)+1;
        PUSH(IPOFF); IP += ALIGN2(sz)),
    CODE("dotstr",
        char *s  = (char*)IP;                     // get string pointer
        int  sz  = STRLENB(s)+1;
        fout << s;  IP += ALIGN2(sz)),            // send to output console
    CODE("branch" , IP = JMPIP),                           // unconditional branch
    CODE("0branch", IP = POP() ? IP + sizeof(IU) : JMPIP), // conditional branch
    CODE("donext",
         if ((rs[-1] -= 1) >= 0) IP = JMPIP;       // rs[-1]-=1 saved 200ms/1M cycles
         else { IP += sizeof(IU); rs.pop(); }),
    CODE("does",                                   // CREATE...DOES... meta-program
         IU *ip  = (IU*)mmu.pfa(WP);
         IU *ipx = (IU*)((U8*)ip + mmu[WP].plen);          // range check
         while (ip < ipx && mmu.ri(ip) != DOES) ip++;      // find DOES
         while (++ip < ipx) add_iu(mmu.ri(ip));            // copy&paste code
         IP = (U8*)ipx),                                   // done
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
    CODE("mod",  top =  fmod(ss.pop(), top)),          /// fmod = x - int(q)*y
    CODE("*/",   top =  ss.pop() * ss.pop() / top),
    CODE("/mod",
        DU n = ss.pop(); DU t = top;
        ss.push(fmod(n, t)); top = round(n / t)),
    CODE("*/mod",
        DU n = ss.pop() * ss.pop();  DU t = top;
        ss.push(fmod(n, t)); top = round(n / t)),
    CODE("and",  top = I2DU(INT(ss.pop()) & INT(top))),
    CODE("or",   top = I2DU(INT(ss.pop()) | INT(top))),
    CODE("xor",  top = I2DU(INT(ss.pop()) ^ INT(top))),
    CODE("abs",  top = abs(top)),
    CODE("negate", top = -top),
    CODE("max",  DU n=ss.pop(); top = (top>n)?top:n),
    CODE("min",  DU n=ss.pop(); top = (top<n)?top:n),
    CODE("2*",   top *= 2),
    CODE("2/",   top /= 2),
    CODE("1+",   top += 1),
    CODE("1-",   top -= 1),
    /// @}
    /// @defgroup Floating Point Math ops
    /// @{
    CODE("int",  top = floor(top)),
    CODE("round",top = INT(top)),
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
    CODE("base@",   PUSH(radix)),
    CODE("base!",   fout << setbase(radix = POPi)),
    CODE("hex",     fout << setbase(radix = 16)),
    CODE("decimal", fout << setbase(radix = 10)),
    CODE("cr",      fout << ENDL),
    CODE(".",       fout << POP() << ' '),
    CODE(".r",      IU n = POPi; dot_r(n, POP())),
    CODE("u.r",     IU n = POPi; dot_r(n, abs(POP()))),
    CODE(".f",      IU n = POPi; fout << setprec(n) << POP()),
    CODE("key",     PUSH(next_word()[0])),
    CODE("emit",    fout << (char)POPi),
    CODE("space",   fout << ' '),
    CODE("spaces",
         int n = POPi;
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
        add_str(DOSTR, s)),                 // dostr, (+parameter field)
    IMMD(".\"",
        const char *s = scan('"')+1;        // string skip first blank
        add_str(DOTSTR, s)),                // dotstr, (+parameter field)
    /// @}
    /// @defgroup Branching ops
    /// @brief - if...then, if...else...then
    /// @{
    IMMD("if", add_iu(ZBRAN); PUSH(LWIP); add_iu(0)),       // if   ( -- here )
    IMMD("else",                                            // else ( here -- there )
        add_iu(BRAN);
        IU h = LWIP;                                        // get current ip address
        add_iu(0); mmu.setjmp(POPi); PUSH(h)),              // set forward jump
    IMMD("then", mmu.setjmp(POPi)),                         // backfill jump address
    /// @}
    /// @defgroup Loops
    /// @brief  - begin...again, begin...f until, begin...f while...repeat
    /// @{
    IMMD("begin",   PUSH(LWIP)),
    IMMD("again",   add_iu(BRAN);  add_iu(POPi)),           // again    ( there -- )
    IMMD("until",   add_iu(ZBRAN); add_iu(POPi)),           // until    ( there -- )
    IMMD("while",   add_iu(ZBRAN); PUSH(LWIP); add_iu(0)),  // while    ( there -- there here )
    IMMD("repeat",  add_iu(BRAN);                           // repeat    ( there1 there2 -- )
        IU t=POPi; add_iu(POPi); mmu.setjmp(t)),            // set forward and loop back address
    /// @}
    /// @defgrouop For loops
    /// @brief  - for...next, for...aft...then...next
    /// @{
    IMMD("for" ,    add_iu(TOR); PUSH(LWIP)),               // for ( -- here )
    IMMD("next",    add_iu(DONEXT); add_iu(POPi)),          // next ( here -- )
    IMMD("aft",                                             // aft ( here -- here there )
        POP(); add_iu(BRAN);
        IU h=LWIP; add_iu(0); PUSH(LWIP); PUSH(h)),
    /// @}
    /// @defgrouop Compiler ops
    /// @{
    CODE(":", mmu.colon(next_word()); compile=true),
    IMMD(";", compile = false),
    CODE("variable",                                        // create a variable
        mmu.colon(next_word());                             // create a new word on dictionary
        add_iu(DOVAR);                                      // dovar (+parameter field)
        add_du(0)),                                         // data storage (32-bit integer now)
    CODE("constant",                                        // create a constant
        mmu.colon(next_word());                             // create a new word on dictionary
        add_iu(DOLIT);                                      // dovar (+parameter field)
        add_du(POP())),                                     // data storage (32-bit integer now)
    /// @}
    /// @defgroup metacompiler
    /// @brief - dict is directly used, instead of shield by macros
    /// @{
    CODE("exit",  IP = mmu.pfa(WP) + mmu[WP].plen),         // quit current word execution
    CODE("exec",  call(POPi)),                              // execute word
    CODE("create",
        mmu.colon(next_word());                             // create a new word on dictionary
        add_iu(DOVAR)),                                     // dovar (+ parameter field)
    CODE("to",              // 3 to x                       // alter the value of a constant
        int w = FIND(next_word());                          // to save the extra @ of a variable
        mmu.wd((DU*)(mmu.pfa(w) + sizeof(IU)), POP())),
    CODE("is",              // ' y is x                     // alias a word
        int w = FIND(next_word());                          // can serve as a function pointer
        mmu.wi((IU*)mmu.pfa(POP()), mmu[w].pidx)),          // but might leave a dangled block
    CODE("[to]",            // : xx 3 [to] y ;              // alter constant in compile mode
        IU w = mmu.ri((IU*)IP); IP += sizeof(IU);           // fetch constant pfa from 'here'
        mmu.wd((DU*)(mmu.pfa(w) + sizeof(IU)), POP())),
    ///
    /// be careful with memory access, especially BYTE because
    /// it could make access misaligned which slows the access speed by 2x
    ///
    CODE("@",     IU w = POPi; PUSH(mmu.rd(w))),                                   // w -- n
    CODE("!",     IU w = POPi; mmu.wd(w, POP())),                                  // n w --
    CODE(",",     DU n = POP(); add_du(n)),
    CODE("allot", DU v = 0; for (IU n = POPi, i = 0; i < n; i++) add_du(v)),       // n --
    CODE("+!",    IU w = POPi; mmu.wd(w, mmu.rd(w)+POP())),                        // n w --
    CODE("?",     IU w = POPi; fout << mmu.rd(w) << " "),                          // w --
    /// @}
    /// @defgroup Debug ops
    /// @{
    CODE("here",  PUSH(mmu.here())),
    CODE("ucase", ucase = POPi),
    CODE("'",     int w = FIND(next_word()); PUSH(w)),
    CODE(".s",    ss_dump(POPi)),
    CODE("words", fout << opx(OP_WORDS)),
    CODE("see",   int w = FIND(next_word()); fout << opx(OP_SEE, (IU)w)),
    CODE("dump",  IU n = POPi; IU a = POPi; fout << opx(OP_DUMP, a, n)),
    CODE("forget",
        int w = FIND(next_word());
        if (w<0) return;
        IU b = FIND("boot")+1;
        mmu.clear(w > b ? w : b)),
    /// @}
    /// @defgroup System ops
    /// @{
    CODE("peek",  IU a = POPi; PUSH(PEEK(a))),
    CODE("poke",  IU a = POPi; POKE(a, POPi)),
    CODE("clock", PUSH(millis())),
    CODE("delay", delay(POPi)),                                // TODO: change to VM_WAIT
    CODE("bye",   status = VM_STOP),
    CODE("boot",  mmu.clear(FIND("boot") + 1))
    /// @}
    };
    for (int i=0; i<sizeof(prim)/sizeof(Code); i++) {
        mmu << (Code*)&prim[i];
        printf("%3d> %p %s\n", i, mmu[i].name, mmu[i].name);   // dump dictionary from device
    }
    status = VM_RUN;

    printf("init() VM=%p sizeof(Code)=%d\n", this, (int)sizeof(Code));
};
///
/// ForthVM Outer interpreter
/// @brief having outer() on device creates branch divergence but
///    + can enable parallel VMs (with different tasks)
///    + can support parallel find()
///    + can support system without a host
///
__GPU__ void
ForthVM::outer() {
    while (fin >> idiom) {                   /// loop throught tib
        printf("%d>> %s => ", blockIdx.x, idiom);
        int w = FIND(idiom);                 /// * search through dictionary
        if (w>=0) {                          /// * word found?
            printf("%p %s %d\n", mmu[w].xt, mmu[w].name, w);
            if (compile && !mmu[w].immd) {   /// * in compile mode?
                add_iu((IU)w);               /// * add found word to new colon word
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
        printf("%f\n", n);
        if (compile) {                       /// * add literal when in compile mode
            add_iu(DOLIT);                   ///> dovar (+parameter field)
            add_du(n);                       ///> store literal
        }
        else PUSH(n);                        ///> or, add value onto data stack
    }
    if (!compile) ss_dump(ss.idx);
}
//=======================================================================================
