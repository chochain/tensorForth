/** -*- c++ -*-
 * @file
 * @brief TensorVM class - extend ForthVM, handle tensor ops implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tenvm.h"

#if T4_ENABLE_OBJ
///
/// override with tensor handler
///
__GPU__ int
TensorVM::process(char *idiom) {
    state = QUERY;
    IU w = parse(idiom);                      /// * parse it as a word
    if (w) return 1;                          /// * success, done
    
    char *p;
    DU n = number(idiom, &p);                 /// * parse it as a literal number
    if (*p!='\0') return 0;                   /// * failed, bail

    SCALAR(n);                                /// * mask out object bit
    if (compile) {                            /// * add literal when in compile mode
        VLOG2("%d> %g\n", id, n);
        add_lit(n);                           ///> dovar (+parameter field)
    }
    else if (ten_lvl > 0) {                   /// * append literal into tensor storage
        VLOG2("%d> T[%d]=%g\n", id, ten_off, n);
        TTOS.data[ten_off++] = n;             /// * append to tensor.data (no stack used)
    }
    else {                                    ///> or, add value onto data stack
        VLOG2("%d> ss.push(%g)=%08x\n", id, n, DU2X(n));
        PUSH(n);
    }
    return 1;
}
///
/// 1-operand self math ops (destructive)
///
__GPU__ void
TensorVM::xop1(math_op op, DU v) {
    ///
    /// scalar handler
    ///
    if (!IS_OBJ(tos)) {                     /// * scalar value
        switch (op) {
        case ABS:  tos = ABS(tos);          break;
        case NEG:  tos = NEG(tos);          break;
        case EXP:  tos = EXP(tos);          break;
        case LN:   tos = LN(tos);           break;
        case LOG:  tos = LOG(tos);          break;
        case TANH: tos = TANH(tos);         break;
        case RELU: tos = MAX(tos, DU0);     break;
        case SIGM: tos = SIGMOID(tos);      break;
        case SQRT: tos = SQRT(tos);         break;
        case RCP:  tos = RCP(tos);          break;
        case SAT:  tos = SAT(tos);          break;
        case POW:  tos = POW(tos, v);       break;
        }
        SCALAR(tos);
        return;
    }
    ///
    /// single tensor handler (destructive)
    ///
    Tensor &A = TTOS;
    if (!A.is_tensor()) { ERROR("tensor?"); return; }
    
    switch (op) {        /// * defined in ~/src/util.h
    case ABS:
    case NEG:
    case EXP:
    case LN:
    case LOG:
    case TANH:
    case RELU:
    case SIGM:
    case SQRT:
    case RCP:
    case SAT:
    case FILL:
    case GFILL:
    case SCALE:
    case POW:   A.map(op, v);   break;
    case IDEN:  A.identity();   break;
    default: ERROR("tenvm#xop1(%d) not supprted\n", op);
    }
}
///
/// 2-operand tensor ops
///
__GPU__ void
TensorVM::xop2(math_op op, t4_drop_opt x) {
    static const char *opn[] = { MATH_OP };
    ///
    /// 2-operand operator (broadcasting)
    ///
    bool s0 = !IS_OBJ(tos), s1 = !IS_OBJ(ss[-1]); /// * scalar flags
    if (s0 && s1) return _ss_op(op);              /// * scalar scalar op
    if (s0) {                                     /// * tensor scaler op
        Tensor &O = _ts_op(op, x);
        VLOG2("tenvm# A[%d,%d] %s %g => O[%d,%d]\n",
              TNOS.H(), TNOS.W(), opn[op], tos, O.H(), O.W());
        if (x==T_KEEP) PUSH(O);
        else           POP();
        return;
    }
    if (s1) {                                     /// * scalar tensor op
        Tensor &O = _st_op(op, x);
        VLOG2("tenvm# %g %s A[%d,%d] => O[%d,%d]\n",
              ss[-1], opn[op], TTOS.H(), TTOS.W(), O.H(), O.W());
        if (x==T_KEEP) PUSH(O);
        else           ss.pop();
        return;
    }

    Tensor &O = _tt_op(op);                       /// * tensor tensor op
    if (O != TTOS) {
        VLOG2("tenvm# A[%d,%d] %s B[%d,%d] => O[%d,%d]\n",
             TNOS.H(), TNOS.W(), opn[op], TTOS.H(), TTOS.W(), O.H(), O.W());
        if (x==T_DROP) {
            DROP(POP());
            DROP(POP());
        }
        PUSH(O);
    }
}
///
/// 1-operand ops with new tensor created (on TOS)
///
__GPU__ void
TensorVM::xop1t(t4_ten_op op) {
    Tensor &A  = TTOS;
    if (!A.is_tensor() || A.rank != 2) { ERROR("tensor2?"); return; }
    ///
    /// single tensor handler
    ///
    Tensor &T = (op == T_INV) ? A : COPY(A);  /// * hardcopy original matrix if needed
    bool   tos = true;
    switch (op) {
    case T_INV: {
        Tensor &I = _tinv(A);                 /// * inverse A matrix
        PUSH(I);                              /// * put on TOS
        tos = false;                          /// * _tinv create its own temp
    } break;
    case T_DET: {                             /// * TODO: use PLU
        int    ns;                            ///> number of row flipping
        Tensor &P = mmu.tensor(A.H());        /// * dummy vector
        Tensor::plu(T, P, &ns);               /// * decompose A to PLU
        DU     v  = T.det();                  /// * multiply diagnal
        PUSH(ns&1 ? -v : v);                  /// * return determinant on TOS
        FREE(P);
        FREE(T);                          /// * not needed
        tos = false;
    } break;
    case T_LU:  Tensor::lu(T);    break;      /// * decompose A to LU
    case T_LUINV:
        Tensor::lu(T);                        /// * create the LU matrix
        Tensor::lu_inverse(T);    break;      /// * inverse it 
    case T_TRIU: T.triu();        break;
    case T_TRIL: T.tril();        break;
    case T_XPOS:
        T.reshape(A.W(), A.H());
        Tensor::transpose(A, T);  break;
    default:
        ERROR("tenvm#xop1t(%d) not supported\n", op);
        FREE(T);
        tos = false;
    }
    if (tos) PUSH(T);
}
///
/// 2-operand tensor ops
///
__GPU__ void
TensorVM::xop2t(t4_ten_op op, t4_drop_opt x) {
    if (!TOS2T) {
        ERROR("tenvm#xop2t TNOS TTOS required!\n");
        return;
    }
    Tensor &A = TNOS, &B = TTOS;
    switch (op){
    case T_DOT: {               ///< C = A @ B
        Tensor &C = _tdot(A, B);
        if (C != B && C != A) {
            if (x==T_DROP) {
                DROP(POP());
                DROP(POP());
            }
            PUSH(C);
        }
    } break;
    case T_DIV: {               ///< C = A @ inverse(B)
        Tensor &C = _tdiv(A, B);
        if (C != B) PUSH(C);
    } break;
    case T_SOLV: {              ///< solve B = AX
        Tensor &X = _solv(A, B);
        PUSH(X);
    } break;
    default:
        ERROR("tenvm#xop2t(%d) not supported\n", op);
    }
}
///
/// scalar-scalar ops
///
__GPU__ __INLINE__ void
TensorVM::_ss_op(math_op op) {               ///< scalar-scalar ops
    switch (op) {
    case ADD:  tos = ADD(ss.pop(), tos); break;
    case SUB:  tos = SUB(ss.pop(), tos); break;
    case MUL:  tos = MUL(ss.pop(), tos); break;
    case DIV:  tos = DIV(ss.pop(), tos); break;
    case MOD:  tos = MOD(ss.pop(), tos); break;
    case MAX:  tos = MAX(ss.pop(), tos); break;
    case MIN:  tos = MIN(ss.pop(), tos); break;
    case MUL2: tos = MUL2(ss.pop(), tos); break;
    case MOD2: tos = MOD2(ss.pop(), tos); break;
    }
    SCALAR(tos);                              /// * even +- can set LSB (rounding)
}

__GPU__ __INLINE__ Tensor&
TensorVM::_st_op(math_op op, t4_drop_opt x) { ///< scalar tensor op
    Tensor &A = TTOS;                         /// * Tensor on TOS
    DU     v  = ss[-1];                       /// * scalar as NOS
    Tensor &O = x==T_KEEP ? COPY(A) : A;      /// * make a hard copy (and parameters)
    if (op==DIV || op==SUB) {                 /// * op(scaler, tensor)
        Tensor &B = mmu.tensor(A.numel);      /// * working tensor
        B.map(FILL, v);                       /// * broadcast
        Tensor::ten_op(op, B, A, O);          /// * Hadamard ops
        FREE(B);                              /// * free working tensor
    }
    else Tensor::ten_op(op, A, v, O);         /// * broadcast_op(tensor, scalar)
    
    return O;
}

__GPU__ __INLINE__ Tensor&
TensorVM::_ts_op(math_op op, t4_drop_opt x) { ///< tensor scalar op
    Tensor &A = TNOS;                         ///< tensor on NOS
    Tensor &O = x==T_KEEP ? COPY(A) : A;      ///< make a hard copy of A
    Tensor::ten_op(op, A, tos, O);            /// * broadcast_op(tensor, scalar)
    return O;
}
///
/// op(tensor, tensor)
///
/**
  In NumPy, the behavior depends on the dimensionality of the Tensors as follows:
  - DONE: If both arguments are 2-dimensional, the matrix-matrix product is returned.
  - DONE: If both Tensors are 1-dimensional, the dot product (scalar) is returned.
  - TODO: If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
  - TODO: If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
  - TODO: If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned.  
    - If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after.  
    - If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. 
    - The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).  
    - For example, if tensor1 is a (j x 1 x n x m) Tensor and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
__GPU__ __INLINE__ Tensor&
TensorVM::_tt_op(math_op op) {                ///< tensor-tensor ops
    Tensor &A = TNOS, &B = TTOS;
    ///
    /// tensor, tensor op
    ///
    if (!A.is_same_shape(B)) return (ERROR("dim?\n"), B);

    Tensor &O = COPY(A);                      ///< make a hard copy
    Tensor::ten_op(op, A, B, O);              /// * Hadamard ops
    if (A.rank==1) O.reshape(O.numel);
    
    return O;
}

__GPU__ Tensor&
TensorVM::_tinv(Tensor &A) {                 ///< matrix inverse
    Tensor &I = mmu.tensor(A.H(), A.W()).identity();
    Tensor &X = COPY(A);                     ///< hardcopy temp, keep A untouched
    Tensor::inverse(X, I);
    FREE(X);                                 /// * release temp 
    return I;
}

__GPU__ __INLINE__ Tensor&
TensorVM::_tdiv(Tensor &A, Tensor &B) {      ///< tensor division
    U16 m = A.H(), ka = A.W(), kb = B.H(), n = B.W();
    if (kb != n || ka != kb) return B;       /// * B square?

    Tensor &I = _tinv(B);
    Tensor &O = mmu.tensor(m, n);
    Tensor::mm(A, I, O);                     /// A * B^-1
    FREE(I);
    
    return O;
}

__GPU__ __INLINE__ Tensor&
TensorVM::_tdot(Tensor &A, Tensor &B) {      ///< A x B tensor dot product
    if (B.rank==1 &&                         ///> dot(vector, vector)
        A.rank==1 && A.numel==B.numel) {
        DU v = A.dot(B);
        PUSH(v);
        VLOG1("tenvm# A[%d] · B[%d] => %g\n", A.H(), B.H(), v);
        return B;                            /// * non-tensor
    }
    if (B.rank==1 && A.W()==B.numel) {       ///> inner(tensor, vector)
        Tensor &O = mmu.tensor(A.H());
        Tensor::mm(A, B, O);
        return O;
    }
    if (A.W()==B.H()) {                      /// * tensor @ tensor
        Tensor &O = mmu.tensor(A.H(), B.W());
        Tensor::mm(A, B, O);
        return O;
    }
    ERROR("A.W!=B.H dim?");
    
    return A;                                /// * i.e. skip in xop2
}

__GPU__ __INLINE__ Tensor&
TensorVM::_solv(Tensor &B, Tensor &A) {      /// Note: A, B flipped 
    U16 m = A.H(), k = A.W(), n = B.H();     /// B[3,1] = A[3,3] * X
    VLOG1("tenvm# solv B[%d] = [%d,%d]*X\n", n, m, k);
    
    if (B.rank!=1 || m!=k || k!=n) return B;
    
    Tensor &I = _tinv(A);
    Tensor &O = mmu.tensor(k);               /// resultant vector
    Tensor::mm(I, B, O);                     /// O = A^-1 x B
    FREE(I);
    
    return O;
}

__GPU__ __INLINE__ void
TensorVM::_gemm() {                          ///< blas GEMM
    if (!TOS3T) { ERROR("tensors?"); return; }
    
    Tensor &O = TTOS, &B = TNOS, &A = (Tensor&)mmu.du2obj(ss[-2]);
    DU     b  = ss[-3];
    DU     a  = ss[-4];
    U16    m  = A.H(), k = A.W(), n = B.W();
    if (k == B.H() && m == O.H() && n == O.W()) {
        Tensor &X = COPY(O);                 /// * hard copy O tensor
        Tensor::gemm(A, B, X, a, b);
        PUSH(X);
    }
    else ERROR("dim?");
}

__GPU__ void
TensorVM::_tprint(DU v) {
    sys.dot(DOT, v);                          /// * send v to output stream
    if (IS_OBJ(v)) {
        mmu.mark_free(v);                     /// * mark to release by host
        state = HOLD;                         /// * forced flush (wasteful but no dangling objects)
    }
}    
__GPU__ void
TensorVM::_pickle(bool save) {
    U8   mode= FAM_WO;                        ///< file mode (W/O,R/W)|BIN
    
    if (ss.idx > 1 && IS_OBJ(ss[-2])) { /* OK */ }
    else if (ss.idx > 2 && IS_OBJ(ss[-3])) mode |= POPi;
    else { ERROR("tensor adr len [mode]?\n"); return; }
    
    IU   len  = POPi;                         ///< string length (not used for now)
    IU   adr  = POPi;                         ///< address to pmem
    char *fn  = (char*)MEM(adr);              ///< pointer to string on PAD
    
    sys.op(OP_TSAVE, mode, tos);              /// * issue save command
    sys.op_fn(fn);                            /// * append filename
    state = HOLD;                             /// * return to CPU
}
///
/// Tensor Vocabulary
///
__GPU__ void
TensorVM::init() {
    if (id !=0) return;                       /// * only needed once
    ForthVM::init();
    ///
    ///@defgroup Tensor creation ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("vector",                            ///< allocate a vector
         IU sz = POPi;
         PUSH(mmu.tensor(sz)));
    CODE("matrix",                            ///< allocate a matrix
         IU w = POPi; IU h = POPi;
         PUSH(mmu.tensor(h, w)));
    CODE("tensor",                            ///< allocate a NHWC tensor
         IU c = POPi; IU w = POPi; IU h = POPi; IU n = POPi;
         PUSH(mmu.tensor(n, h, w, c)));
    CODE("vector{",                           ///< create a vector with literals
         IU sz = POPi;
         PUSH(mmu.tensor(sz));
         ten_off = 0; ten_lvl = 1);
    CODE("matrix{",                           ///< create a matrix with literals
         IU w = POPi; IU h = POPi;
         PUSH(mmu.tensor(h, w));
         ten_off = 0; ten_lvl = 1);
    CODE("view",   PUSH(DUP(tos)));           ///< create a view of a tensor
    CODE("copy",   PUSH(COPY(tos)));          ///< create a hardcopy of a tensor
    ///@}
    ///@defgroup Tensor shape ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("flatten",                           ///< reshape as a vector (1-D array)
         Tensor &t = TTOS;
         t.reshape(t.numel));
    CODE("reshape2",                          ///< reshape as matrix(h,w)
         IU w = POPi; IU h = POPi;
         TTOS.reshape(h, w));
    CODE("reshape4",                          ///< reshape as Tensor(NHWC)
         IU c = POPi; IU w = POPi; IU h = POPi; IU n = POPi;
         TTOS.reshape(n, h, w, c));
    CODE("same_shape?",
         if (IS_OBJ(tos) && IS_OBJ(ss[-1])) {
             Tensor &A=TTOS; Tensor &B=TNOS; PUSH(BOOL(A.is_same_shape(B)));
         }
         else ERROR("TOS,NOS tensors?"));
    ///@}
    ///@defgroup Tensor fill ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("={",                                 ///< (n -- ) or ( -- )
         ten_off = IS_OBJ(tos) ? 0 : POPi;
         ten_lvl = IS_OBJ(tos) ? 1 : 0);
    CODE("zeros", xop1(FILL, DU0));            ///< fill tensor with 0s
    CODE("ones",  xop1(FILL, DU1));            ///< fill tensor with 1s
    CODE("full",  xop1(FILL, POP()));          ///< fill tensor with a value
    CODE("gradfill", xop1(GFILL, DU1));        ///< gradient fill a tensor
    CODE("eye",   xop1(IDEN));                 ///< fill 1s in diag
    CODE("rand",  tos = sys.rand(tos, UNIFORM));   ///< uniform randomize a tensor or number
    CODE("randn", tos = sys.rand(tos, NORMAL));    ///< normal dist. randomize a tensor
    ///@}
    ///@defgrup Tensor slice and dice
    ///@{
    CODE("normalize",
         DU std = POP(); DU avg = POP();
         if (TOS1T) { TTOS.normalize(std, avg); });
    CODE("sum", if (TOS1T) PUSH(TTOS.sum()));
    CODE("avg", if (TOS1T) PUSH(TTOS.avg()));
    CODE("std", if (TOS1T) PUSH(TTOS.std()));
    CODE("{",   if (TOS1T && ten_lvl > 0) ++ten_lvl);
    CODE("}",   if (TOS1T && ten_lvl > 0) --ten_lvl);
    CODE("slice",
         IU y1 = POPi; IU y0 = POPi; IU x1 = POPi; IU x0 = POPi;
         if (TOS1T) {
             Tensor &t0 = TTOS;
             Tensor &t1 = mmu.slice(t0, x0, x1, y0, y1);
             PUSH(t1);
         });
    CODE("t@", 
         if (!IS_OBJ(ss[-1]) && IS_OBJ(tos)) {
             IU i = POPi; DU v = TTOS[i];
             SCALAR(v);
             PUSH(v);
         });
    CODE("t!",  DU v = POP(); IU i = POPi; if (IS_OBJ(tos)) TTOS[i]=v);
    ///@}
    ///@defgroup 1-tensor ops in-place (i.e. destructive, as in Forth)
    ///@{
    CODE("exp",       xop1(EXP));             ///< (A -- A')
    CODE("ln",        xop1(LN));                      
    CODE("log",       xop1(LOG));                     
    CODE("tanh",      xop1(TANH));
    CODE("relu",      xop1(RELU));
    CODE("sigmoid",   xop1(SIGM));
    CODE("sqrt",      xop1(SQRT));
    CODE("1/x",       xop1(RCP));             ///< reciprocal
    CODE("sat",       xop1(SAT));
    CODE("pow",       xop1(POW, POP()));      ///< scale tensor with TOS
    ///@}
    ///@defgroup 1-tensor ops that create new tensor
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("inverse",   xop1t(T_INV));          ///< (A -- A Ai')   matrix inversion (GaussJordan)
    CODE("det",       xop1t(T_DET));          ///< (A -- A d)     matrix determinant
    CODE("lu",        xop1t(T_LU));           ///< (A -- A A')    LU decomposition
    CODE("luinv",     xop1t(T_LUINV));        ///< (A -- A A')    inverse the LU matrix
    CODE("upper",     xop1t(T_TRIU));         ///< (A -- A A')    upper triangle
    CODE("lower",     xop1t(T_TRIL));         ///< (A -- A A')    lower triangle
    CODE("transpose", xop1t(T_XPOS));         ///< (A -- A At)    matrix transpose
    ///@}
    ///@defgroup 2-tensor matrix ops
    ///@{
    CODE("+=",        xop2(ADD, T_DROP));     ///< (A B -- C)
    CODE("-=",        xop2(SUB, T_DROP));
    CODE("*=",        xop2(MUL, T_DROP));
    CODE("/=",        xop2(DIV, T_DROP));
    CODE("@=",        xop2t(T_DOT, T_DROP));  ///< (A B -- C)
    CODE("matmul",    xop2t(T_DOT));          ///< (A B -- A B C) matrix multiply
    CODE("matdiv",    xop2t(T_DIV));          ///< (A B -- A B C) matrix divide
    CODE("solve",     xop2t(T_SOLV));         ///< (B A -- B A X) solve B = AX
    CODE("gemm",      _gemm());               ///< (a b A B C -- a b A B C') GEMM (C updated)
    ///@}
    ///@defgroup Tensor persistance
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("bin",       PUSH(FAM_RAW));         ///< raw/binary file
    CODE("w/o",       PUSH(FAM_WO));          ///< write-only file
    CODE("r/w",       PUSH(FAM_RW));          ///< read-write file
    CODE("save",      _pickle(true));         ///< ( T fn len -- T ) save tensor to a file
    CODE("load",      _pickle(false));        ///< ( T fn -- T' ) fill a tensor from file
    ///
    /// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ///
    ///@defgroup redefined tensor ops
    ///@{
    CODE("boot",      mmu.clear(FIND((char*)"load") + 1));
    CODE(".",         _tprint(POP()));           ///< print TOS
    CODE("+",         xop2(ADD, T_KEEP));
    CODE("-",         xop2(SUB, T_KEEP));
    CODE("*",         xop2(MUL, T_KEEP));
    CODE("/",         xop2(DIV, T_KEEP));
    CODE("abs",       xop1(ABS));
    CODE("negate",    xop1(NEG));
    CODE("@",
         if (TOS2T) xop2t(T_DOT);             ///< matrix @ product
         else {
             DU v = mmu.rd(POPi);
             PUSH(DUP(v));
         });
    CODE("max",
         if (IS_OBJ(tos)) PUSH(TTOS.max());
         else xop2(MAX));
    CODE("min",
         if (IS_OBJ(tos)) PUSH(TTOS.min());
         else xop2(MIN));
    ///@}
    TRACE("TensorVM[%d]::init ok, sizeof(Tensor)=%ld\n", id, sizeof(Tensor));
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
