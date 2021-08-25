#include <iomanip>          // setbase, setw, setfill
#include "ceforth.h"

__GPU__ int code_fence = 0, code_ip = 0;
///
/// Code class constructors
///
__GPU__ Code::Code(string n, fop fn, bool im) {
	name = n; token = code_fence++; immd = im; xt = fn;
}
__GPU__ Code::Code(string n, bool f)   { name = n; if (f) token = code_fence++; }
__GPU__ Code::Code(Code *c, DTYPE v)   { name = c->name; xt = c->xt; qf.push(v); }
__GPU__ Code::Code(Code *c, string s)  { name = c->name; xt = c->xt; if (s!=string()) literal = s;  }

__GPU__ Code*  Code::addcode(Code* w)  { pf.push(w);   return this; }
__GPU__ string Code::to_s()    { return name + " " + to_string(token) + (immd ? "*" : ""); }
__GPU__ string Code::see(int dp) {
    stringstream cout("");
    auto see_pf = [&cout](int dp, string s, ForthList<Code*> &a) {   // lambda for indentation and recursive dump
        int i = dp; cout << ENDL; while (i--) cout << "  "; cout << s;
        for (int i=0; i<a.size(); i++) cout << a[i]->see(dp + 1);
    };
    auto see_qf = [&cout](ForthList<DTYPE> &a) {
    	cout << " = "; for (int i=0; i<a.size(); i++) cout << a[i] << " ";
    };
    see_pf(dp, "[ " + to_s(), pf);
    if (pf1.size() > 0) see_pf(dp, "1--", pf1);
    if (pf2.size() > 0) see_pf(dp, "2--", pf2);
    if (qf.size()  > 0) see_qf(qf);
    cout << "]";
    return cout.str();
}
__GPU__ void Code::nest() {
    if (xt) xt(this);
    else {
    	int tmp = code_ip; code_ip = 0;
        for (Code* w : pf.v) { yield(); w->nest(); code_ip++; } /// run inner interpreter
        code_ip = tmp;
    }
}
///
/// ForthVM class constructor
///
__GPU__ ForthVM::ForthVM(istream &in, ostream &out) : cin(in), cout(out) {}
///
/// dictionary and input stream search functions
///
__GPU__ __INLINE__ DTYPE ForthVM::POP()         { DTYPE n = top; top = ss.pop(); return n; }
__GPU__ __INLINE__ DTYPE ForthVM::PUSH(DTYPE v) { ss.push(top); return top = v; }

/// search dictionary reversely
__GPU__ Code *ForthVM::find(string s) {
    for (int i = dict.size() - 1; i >= 0; --i) {
        if (s == dict[i]->name) return dict[i];
    }
    return NULL;
}
__GPU__ string ForthVM::next_idiom(char delim) {
    string s; delim ? getline(cin, s, delim) : cin >> s; return s;
}
__GPU__ void ForthVM::dot_r(int n, DTYPE v) {
    cout << setw(n) << setfill(' ') << v;
}
__GPU__ void ForthVM::ss_dump() {
    cout << " <"; for (DTYPE i : ss.v) { cout << i << " "; }
    cout << top << "> ok" << ENDL;
}
__GPU__ void ForthVM::words() {
    int i = 0;
    for (Code* w : dict.v) {
        if ((i++ % 10) == 0) { cout << ENDL; yield(); }
        cout << w->to_s() << " ";
    }
}
__GPU__ void ForthVM::call(Code *w) {
    int tmp = WP;                                       /// * setup call frame
    WP = w->token;
    w->nest(); 											/// * run inner interpreter recursively
/*
    try { w->nest(); }
    catch (exception& e) {
        string msg = e.what();                          /// * capture exception message
        if (msg != string()) cout << msg << ENDL;
    }
*/
    WP = tmp;                                           /// * restore call frame
    yield();
}
///
/// macros to reduce verbosity (but harder to single-step debug)
///
#define CODE(s, g) new Code(string(s), [this](Code *c){ g; })
#define IMMD(s, g) new Code(string(s), [this](Code *c){ g; }, true)
#define INT(f)         (static_cast<int>(f))
#define ALU(a, OP, b)  (INT(a) OP INT(b))
#define BOOL(f) ((f) ? -1 : 0)
///
/// dictionary initializer
///
__GPU__ void ForthVM::init() {
    const Code* prim[] = {       /// singleton, build once onl
    ///
    /// @defgroup Stack op
    /// @{
    CODE("dup",  PUSH(top)),
    CODE("drop", top = ss.pop()),
    CODE("over", PUSH(ss[-1])),
    CODE("swap", DTYPE n = ss.pop(); PUSH(n)),
    CODE("rot",  DTYPE n = ss.pop(); DTYPE m = ss.pop(); ss.push(n); PUSH(m)),
    CODE("-rot", DTYPE n = ss.pop(); DTYPE m = ss.pop(); PUSH(n); PUSH(m)),
    CODE("pick", DTYPE i = top; top = ss[-i]),
    CODE(">r",   rs.push(POP())),
    CODE("r>",   PUSH(rs.pop())),
    CODE("r@",   PUSH(rs[-1])),
    /// @}
    /// @defgroup Stack ops - double
    /// @{
    CODE("2dup", PUSH(ss[-1]); PUSH(ss[-1])),
    CODE("2drop",ss.pop(); top = ss.pop()),
    CODE("2over",PUSH(ss[-3]); PUSH(ss[-3])),
    CODE("2swap",
        DTYPE n = ss.pop(); DTYPE m = ss.pop(); DTYPE l = ss.pop();
        ss.push(n); PUSH(l); PUSH(m)),
    /// @}
    /// @defgroup ALU ops
    /// @{
    CODE("+",    top += ss.pop()),
    CODE("-",    top =  ss.pop() - top),
    CODE("*",    top *= ss.pop()),
    CODE("/",    top =  ss.pop() / top),
    CODE("mod",  top = ALU(ss.pop(), %, top)),
    CODE("*/",   top = ss.pop() * ss.pop() / top),
    CODE("*/mod",
        int n = INT(ss.pop() * ss.pop());
        int t = INT(top);
        ss.push(n % t); top = (n / t)),
    CODE("and",  top = ALU(ss.pop(), &, top)),
    CODE("or",   top = ALU(ss.pop(), |, top)),
    CODE("xor",  top = ALU(ss.pop(), ^, top)),
    CODE("negate", top = -top),
    CODE("abs",  top = abs(top)),
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
    CODE("base!",   cout << setbase(base = POP())),
    CODE("hex",     cout << setbase(base = 16)),
    CODE("decimal", cout << setbase(base = 10)),
    CODE("cr",      cout << ENDL),
    CODE(".",       cout << POP() << " "),
    CODE(".r",      int n = INT(POP()); dot_r(n, POP())),
    CODE("u.r",     int n = INT(POP()); dot_r(n, abs(POP()))),
    CODE(".f",      int n = INT(POP()); cout << setprecision(n) << POP()),
    CODE("key",     PUSH(next_idiom()[0])),
    CODE("emit",    char b = (char)POP(); cout << b),
    CODE("space",   cout << " "),
    CODE("spaces",  for (int n = INT(POP()), i = 0; i < n; i++) cout << " "),
    /// @}
    /// @defgroup Literal ops
    /// @{
    CODE("dotstr",  cout << c->literal),
    CODE("dolit",   PUSH(c->qf[0])),
    CODE("dovar",   PUSH(c->token)),
    CODE("[",       compile = false),
    CODE("]",       compile = true),
    IMMD("(",       next_idiom(')')),
    IMMD(".(",      cout << next_idiom(')')),
    CODE("\\",      cout << next_idiom('\n')),
    CODE("$\"",
        string s = next_idiom('"').substr(1);
        dict[-1]->addcode(new Code(find("dovar"), s))),
    IMMD(".\"",
        string s = next_idiom('"').substr(1);
        dict[-1]->addcode(new Code(find("dotstr"), s))),
    /// @}
    /// @defgroup Branching ops
    /// @brief - if...then, if...else...then
    /// @{
    IMMD("bran",
        bool f = POP() != 0;                        // check flag
        for (Code* w : (f ? c->pf.v : c->pf1.v)) call(w)),
    IMMD("if",
        dict[-1]->addcode(new Code(find("bran")));
        dict.push(new Code("temp"))),               // use last cell of dictionay as scratch pad
    IMMD("else",
        Code *temp = dict[-1]; Code *last = dict[-2]->pf[-1];
        last->pf.merge(temp->pf);
        temp->pf.clear();
        last->stage = 1),
    IMMD("then",
        Code *temp = dict[-1]; Code *last = dict[-2]->pf[-1];
        if (last->stage == 0) {                     // if...then
            last->pf.merge(temp->pf);
            dict.pop();
        }
        else {                                      // if...else...then, or
             last->pf1.merge(temp->pf);             // for...aft...then...next
             if (last->stage == 1) dict.pop();
             else temp->pf.clear();
        }),
    /// @}
    /// @defgroup Loops
    /// @brief  - begin...again, begin...f until, begin...f while...repeat
    /// @{
    CODE("loop",
        while (true) {
            for (Code* w : c->pf.v) call(w);                       // begin...
            int f = INT(top);
            if (c->stage == 0 && (top = ss.pop(), f != 0)) break;  // ...until
            if (c->stage == 1) continue;                           // ...again
            if (c->stage == 2 && (top = ss.pop(), f == 0)) break;  // while...repeat
            for (Code* w : c->pf1.v) call(w);
        }),
    IMMD("begin",
        dict[-1]->addcode(new Code(find("loop")));
        dict.push(new Code("temp"))),
    IMMD("while",
        Code *last = dict[-2]->pf[-1]; Code *temp = dict[-1];
        last->pf.merge(temp->pf);
        temp->pf.clear(); last->stage = 2),
    IMMD("repeat",
        Code *last = dict[-2]->pf[-1]; Code *temp = dict[-1];
        last->pf1.merge(temp->pf); dict.pop()),
    IMMD("again",
        Code *last = dict[-2]->pf[-1]; Code *temp = dict[-1];
        last->pf.merge(temp->pf);
        last->stage = 1; dict.pop()),
    IMMD("until",
        Code *last = dict[-2]->pf[-1]; Code *temp = dict[-1];
        last->pf.merge(temp->pf); dict.pop()),
    /// @}
    /// @defgrouop For loops
    /// @brief  - for...next, for...aft...then...next
    /// @{
    CODE("cycle",
        do { for (Code* w : c->pf.v) call(w); }
        while (c->stage == 0 && rs.dec_i() >= 0);    // for...next only
        while (c->stage > 0) {                       // aft
            for (Code* w : c->pf2.v) call(w);        // then...next
            if (rs.dec_i() < 0) break;
            for (Code* w : c->pf1.v) call(w);        // aft...then
        }
        rs.pop()),
    IMMD("for",
        dict[-1]->addcode(new Code(find(">r")));
        dict[-1]->addcode(new Code(find("cycle")));
        dict.push(new Code("temp"))),
    IMMD("aft",
        Code *last = dict[-2]->pf[-1]; Code *temp = dict[-1];
        last->pf.merge(temp->pf);
        temp->pf.clear(); last->stage = 3),
    IMMD("next",
        Code *last = dict[-2]->pf[-1]; Code *temp = dict[-1];
        if (last->stage == 0) last->pf.merge(temp->pf);
        else last->pf2.merge(temp->pf); dict.pop()),
    /// @}
    /// @defgrouop Compiler ops
    /// @{
    CODE("exec", int n = INT(top); call(dict[n])),
    CODE(":",
        dict.push(new Code(next_idiom(), true));    // create new word
        compile = true),
    IMMD(";", compile = false),
    CODE("variable",
        dict.push(new Code(next_idiom(), true));
        Code *last = dict[-1]->addcode(new Code(find("dovar"), 0));
        last->pf[0]->token = last->token),
    CODE("constant",
        dict.push(new Code(next_idiom(), true));
        Code *last = dict[-1]->addcode(new Code(find("dolit"), POP()));
        last->pf[0]->token = last->token),
    CODE("@",      int w = INT(POP()); PUSH(dict[w]->pf[0]->qf[0])),         // w -- n
    CODE("!",      int w = INT(POP()); dict[w]->pf[0]->qf[0] = POP()),       // n w --
    CODE("+!",     int w = INT(POP()); dict[w]->pf[0]->qf[0] += POP()),      // n w --
    CODE("?",      int w = INT(POP()); cout << dict[w]->pf[0]->qf[0] << " "),// w --
    CODE("array@", int a = INT(POP()); PUSH(dict[INT(POP())]->pf[0]->qf[a])),     // w a -- n
    CODE("array!", int a = INT(POP()); int w = INT(POP());  dict[w]->pf[0]->qf[a] = POP()),   // n w a --
    CODE("allot",                                           // n --
        for (int n = INT(POP()), i = 0; i < n; i++) dict[-1]->pf[0]->qf.push(DVAL)),
    CODE(",",      dict[-1]->pf[0]->qf.push(POP())),
    /// @}
    /// @defgroup metacompiler
    /// @{
    CODE("create",
        dict.push(new Code(next_idiom(), true));            // create a new word
        Code *last = dict[-1]->addcode(new Code(find("dovar"), DVAL));
        last->pf[0]->token = last->token;
        last->pf[0]->qf.clear()),
    CODE("does",
        ForthList<Code*> &src = dict[WP]->pf;               // source word : xx create...does...;
        int i = code_ip; int n = src.size();
        while (++i < n) dict[-1]->pf.push(src[i])),         // copy words after "does" to new the word
    CODE("to",                                              // n -- , compile only
        Code *tgt = find(next_idiom());
        if (tgt) tgt->pf[0]->qf[0] = POP()),                // update constant
    CODE("is",                                              // w -- , execute only
        Code *tgt = find(next_idiom());
        if (tgt) {
            tgt->pf.clear();
            tgt->pf.merge(dict[POP()]->pf);
        }),
    CODE("[to]",
        ForthList<Code*> &src = dict[WP]->pf;               // source word : xx create...does...;
        src[++code_ip]->pf[0]->qf[0] = POP()),              // change the following constant
    /// @}
    /// @defgroup Debug ops
    /// @{
    CODE("bye",   exit(0)),
    CODE("here",  PUSH(dict[-1]->token)),
    CODE("words", words()),
    CODE(".s",    ss_dump()),
    CODE("'",     Code *w = find(next_idiom()); PUSH(w->token)),
    CODE("see",
        Code *w = find(next_idiom());
        if (w) cout << w->see(0) << ENDL),
    CODE("forget",
        Code *w = find(next_idiom());
         if (w == NULL) return;
         dict.clear(code_fence=max(w->token, find("boot")->token + 1))),
    CODE("boot", dict.clear(code_fence=find("boot")->token + 1))
    /// @}
    };
    dict.merge((Code*)prim, code_fence);       /// * populate dictionary
}
///
/// ForthVM Outer interpreter
///
__GPU__ void ForthVM::outer() {
    string idiom;
    while (cin >> idiom) {
        //Serial.print(idiom.c_str()); Serial.print("=>");
    	//printf("%s=>", idiom.c_str());
        Code *w = find(idiom);                          /// * search through dictionary
        if (w) {                                        /// * word found?
            //Serial.println(w->to_s().c_str());
            //printf("%s\n", w->to_s().c_str());
            if (compile && !w->immd)                    /// * in compile mode?
                dict[-1]->addcode(w);                   /// * add to colon word
            else call(w);                               /// * execute forth word
            continue;
        }
        // try as a number
        char *p;
        int n = (int)strtol(idiom.c_str(), &p, base);
        //Serial.println(n, base);
        //printf("%d\n", n);
        if (*p != '\0') {                           /// * not number
            cout << idiom << "? " << ENDL;          ///> display error prompt
            compile = false;                        ///> reset to interpreter mode
            getline(cin, idiom, '\n');              ///> skip the entire line
            continue;
        }
        // is a number
        if (compile)                           /// * a number in compile mode?
            dict[-1]->addcode(new Code(find("dolit"), n)); ///> add to current word
        else PUSH(n);                           	///> or, add value onto data stack
    }
    if (!compile) ss_dump();  /// * dump stack and display ok prompt
}

/// main program
__KERN__ int eforth_init(U8 *cin, U8 *cout) {
	if (threadId.x!=0 || blockId.x!=0) return 0;
    string cmd;

    ForthVM *vm = new ForthVM(forth_in, forth_out);		// create FVM instance
    vm->init();                                 		// initialize dictionary

    while (getline(cin, cmd)) {							// fetch user input
    	//printf("cmd=<%s>\n", line.c_str());
    	forth_in.clear();								// clear any input stream error bit
    	forth_in.str(cmd);								// send command to FVM
        vm->outer();									// execute outer interpreter
        cout << forth_out.str();						// send VM result to output
        forth_out.str(string());						// clear output buffer
    }
    cout << "done!" << ENDL;
    return 0;
}
