/** -*- c++ -*-
 * @file
 * @brief TensorVM class - extend ForthVM, handle tensor ops implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tenvm.h"

#if T4_ENABLE_OBJ
///
/// Tensor print to host
///
__GPU__ void
TensorVM::tprint() {
    if (TOS1T) {                             /// * handle objects
        DU d = POP(); fout << d;
        mmu.mark_free(d);
    }
    else if (IS_OBJ(top)) fout << top;       /// * view, model, dataset
    else fout << " " << POP();               /// * eForth has a space prefix
}
///
/// Tensor-self ops
///
__GPU__ void
TensorVM::xop1(t4_ten_op op, DU v) {
    ///
    /// scalar handler
    ///
    if (!IS_OBJ(top)) {                     /// * scalar value
        switch (op) {
        case O_POW:  top = POW(top, v);       break;
        case O_ABS:  top = ABS(top);          break;
        case O_EXP:  top = EXP(top);          break;
        case O_LOG:  top = LOG(top);          break;
        case O_LN:   top = LN(top);           break;
        case O_TANH: top = TANH(top);         break;
        case O_RELU: top = top > v ? top : v; break;
        case O_SIGM: top = SIGMOID(top);      break;
        }
        SCALAR(top);
        return;
    }
    ///
    /// single tensor handler
    ///
    Tensor &A = TTOS;
    if (!A.is_tensor()) { ERROR("tensor?"); return; }
    
    switch (op) {
    /// ops => update in-place
    case O_FILL:
    case O_SCALE:
    case O_POW:   A.map(op, v);   break;
    case O_ABS:
    case O_EXP:
    case O_LN:
    case O_LOG:
    case O_TANH:
    case O_RELU:
    case O_SIGM:  A.map(op);      break;
    case O_IDEN:  A.identity();   break;
    default: ERROR("TensorVM#xop1(%d) not supprted\n", op);
    }
}
///
/// 1-operand ops with new tensor created (on TOS)
///
__GPU__ void
TensorVM::xop1x(t4_ten_op op) {
    Tensor &A  = TTOS;
    if (!A.is_tensor() || A.rank != 2) { ERROR("tensor2?"); return; }
    ///
    /// single tensor handler
    ///
    Tensor &t = (op == O_INV) ? A : mmu.copy(A); /// * hardcopy original matrix if needed
    bool   tos = true;
    switch (op) {
    case O_INV:
        PUSH(_tinv(A));                       /// * inverse A matrix
        tos = false;             break;       /// * _tinv create its own temp
    case O_DET: {                             /// * TODO: use PLU
        int    ns;                            ///> number of row flipping
        Tensor &P = mmu.tensor(A.H());        /// * dummy vector
        Tensor::plu(t, P, &ns);               /// * decompose A to PLU
        DU     v  = t.det();                  /// * multiply diagnal
        PUSH(ns&1 ? -v : v);                  /// * return determinant on TOS
        mmu.free(P);
        mmu.free(t);                          /// * not needed
        tos = false;
    } break;
    case O_LU:  Tensor::lu(t);    break;      /// * decompose A to LU
    case O_LUINV:
        Tensor::lu(t);                        /// * create the LU matrix
        Tensor::lu_inverse(t);    break;      /// * inverse it 
    case O_TRIU: t.triu();        break;
    case O_TRIL: t.tril();        break;
    case O_XPOS:
        t.reshape(A.W(), A.H());
        Tensor::transpose(A, t);  break;
    default:
        ERROR("TensorVM#xop1x(%d) not supported\n", op);
        mmu.free(t);
        tos = false;
    }
    if (tos) PUSH(t);
}

__GPU__ void
TensorVM::xop2(t4_ten_op op, t4_drop_opt x) {
    static const char *opn[] = { "+", "-", "*", "/", "@", "x" };
    ///
    /// 2-operand operator
    ///
    bool s0 = !IS_OBJ(top), s1 = !IS_OBJ(ss[-1]); /// * scalar flags
    if (s0 && s1) return _ss_op(op);              /// * scalar scalar op
    if (s0) {                                     /// * tensor scaler op
        Tensor &O = _ts_op(op);
        VLOG1("A[%d,%d] %s %f => O[%d,%d]\n",
              TNOS.H(), TNOS.W(), opn[op], top, O.H(), O.W());
        if (x==DROP) { POP(); mmu.free(TTOS); POP(); }
        PUSH(O);
        return;
    }
    if (s1) {                                     /// * scalar tensor op
        Tensor &O = _st_op(op);
        VLOG1("%f %s A[%d,%d] => O[%d,%d]\n",
              ss[-1], opn[op], TTOS.H(), TTOS.W(), O.H(), O.W());
        if (x==DROP) { mmu.free(TTOS); POP(); POP(); }
        PUSH(O);
        return;
    }
    
    Tensor &O = _tt_op(op);                       /// * tensor tensor op
    if (O != TTOS) {
        VLOG1("TensorVM# A[%d,%d] %s B[%d,%d] => O[%d,%d]\n",
             TNOS.H(), TNOS.W(), opn[op], TTOS.H(), TTOS.W(), O.H(), O.W());
        if (x==DROP) { mmu.free(TTOS); POP(); mmu.free(TTOS); POP(); }
        PUSH(O);
    }
}
/**
  TODO: Matrix product of two Tensors.
  The behavior depends on the dimensionality of the Tensors as follows:
  - DONE: If both arguments are 2-dimensional, the matrix-matrix product is returned.
  - DONE: If both Tensors are 1-dimensional, the dot product (scalar) is returned.
  - TODO: If the first argument is 2-dimensional and the second argument is 1-dimensional,
    the matrix-vector product is returned.
  - TODO: If the first argument is 1-dimensional and the second argument is 2-dimensional,
    a 1 is prepended to its dimension for the purpose of the matrix multiply.
    After the matrix multiply, the prepended dimension is removed.
  - TODO: If both arguments are at least 1-dimensional and at least one argument is
    N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
    argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
    batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
    1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
    The non-matrix (i.e. batch) dimensions are broadcasted (and thus
    must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
    and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
__GPU__ void
TensorVM::_ss_op(t4_ten_op op) {               ///< scalar-scalar ops
    switch (op) {
    case O_ADD: top += ss.pop();      break;
    case O_SUB: top = ss.pop() - top; break;
    case O_MUL: top *= ss.pop();      break;
    case O_DIV: top = DIV(ss.pop(), top); break;
    }
    SCALAR(top);                               /// * even +- can set LSB
}

__GPU__ Tensor&
TensorVM::_st_op(t4_ten_op op) {              ///< scalar tensor op
    Tensor &A = TTOS;                         /// * Tensor on TOS
    DU     v  = ss[-1];                       /// * scalar as NOS
    Tensor &O = mmu.copy(A);                  /// * make a hard copy (and parameters)
    if (op==O_DIV || op==O_SUB) {             /// * op(scaler, tensor)
        Tensor &B = mmu.tensor(A.numel);      /// * working tensor
        B.map(O_FILL, v);                     /// * broadcast
        Tensor::ten_op(op, B, A, O);          /// * Hadamard ops
        mmu.free(B);                          /// * free working tensor
    }
    else Tensor::ten_op(op, A, v, O);         /// * broadcast_op(tensor, scalar)
    
    return O;
}

__GPU__ Tensor&
TensorVM::_ts_op(t4_ten_op op) {              ///< tensor scalar op
    Tensor &A = TNOS;                         ///< tensor on NOS
    Tensor &O = mmu.copy(A);                  ///< make a hard copy of A
    Tensor::ten_op(op, A, top, O);            /// * broadcast_op(tensor, scalar)
    
    return O;
}

///
/// op(tensor, tensor)
///
__GPU__ Tensor&
TensorVM::_tt_op(t4_ten_op op) {              ///< tensor-tensor ops
    Tensor &A = TNOS, &B = TTOS;
    ///
    /// broadcast_op(tensor, tensor)
    ///
    if (op == O_DOT)  return _tdot(A, B);    /// * O = A · B
    if (op == O_SOLV) return _solv(B, A);    /// * solve B = A @ X (notation flipped)
    
    if (!A.is_same_shape(B)) return (ERROR("dim?\n"), B);
    
    Tensor &O = mmu.copy(A);                  ///< make a hard copy
    Tensor::ten_op(op, A, B, O);              /// * Hadamard ops
    if (A.rank==1) O.reshape(O.numel);
    
    return O;
}

__GPU__ Tensor&
TensorVM::_tinv(Tensor &A) {                 ///< matrix inverse
    Tensor &I = mmu.tensor(A.H(), A.W()).identity();
    Tensor &X = mmu.copy(A);                 ///< tmep, keep A untouched
    Tensor::inverse(X, I);
    mmu.free(X);                             /// * release temp 
    return I;
}

__GPU__ Tensor&
TensorVM::_tdiv(Tensor &A, Tensor &B) {      ///< tensor division
    U16 m = A.H(), ka = A.W(), kb = B.H(), n = B.W();
    if (kb != n || ka != kb) return B;       /// * B square?

    Tensor &I = _tinv(B);
    Tensor &O = mmu.tensor(m, n);
    Tensor::mm(A, I, O);                     /// A * B^-1
    mmu.free(I);
    
    return O;
}

__GPU__ Tensor&
TensorVM::_tdot(Tensor &A, Tensor &B) {      ///< tensor dot product
    if (B.rank==1 &&                         ///> dot(vector, vector)
        A.rank==1 && A.numel==B.numel) {
        DU v = A.dot(B);
        PUSH(v);
        VLOG1("A[%d] · B[%d] => %f\n", A.H(), B.H(), v);
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
}

__GPU__ Tensor&
TensorVM::_solv(Tensor &A, Tensor &B) {      /// Note: A B flipped [3,3]x[3,1]
    U16 m = A.H(), k = A.W(), n = B.H();
    VLOG1("solv[%d,%d] x [%d]\n", m, k, n);
    
    if (B.rank!=1 || m!=k || k!=n) return A;
    
    Tensor &I = _tinv(A);
    Tensor &O = mmu.tensor(k);               /// resultant vector
    Tensor::mm(I, B, O);                     /// O = A^-1 x B
    mmu.free(I);
    
    return O;
}

__GPU__ void
TensorVM::_gemm() {                          ///< blas GEMM
    if (!TOS3T) { ERROR("tensors?"); return; }
    
    Tensor &O = TTOS, &B = TNOS, &A = (Tensor&)mmu.du2obj(ss[-2]);
    DU     b  = ss[-3];
    DU     a  = ss[-4];
    U16    m  = A.H(), k = A.W(), n = B.W();
    if (k == B.H() && m == O.H() && n == O.W()) {
        Tensor &X = mmu.copy(O);             /// * hard copy O tensor
        Tensor::gemm(A, B, X, a, b);
        PUSH(X);
    }
    else ERROR("dim?");
}

__GPU__ void
TensorVM::_save() {
    IU   mode= FAM_WO | FAM_RAW;            ///< file mode (W/O,R/W)|BIN
    
    if (ss.idx > 1 && IS_OBJ(ss[-2])) { /* OK */ }
    else if (ss.idx > 2 && IS_OBJ(ss[-3])) mode = POPi;
    else { ERROR("tensor adr len [mode]?\n"); return; }
    
    IU   len = POPi;                        ///< string length (not used for now)
    IU   adr = POPi;                        ///< address to pmem
    char *fn = (char*)mmu.pmem(adr);        ///< pointer to string on PAD
    fout << opx(OP_TSAVE, mode, top) << fn; /// * issue save command
    state = VM_WAIT;                        /// * return to CPU
}
///
/// Tensor Vocabulary
///
__GPU__ void
TensorVM::init() {
    const Code prim[] = {                ///< singleton, build once only
    ///@defgroup Tensor creation ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("vector",                       ///< allocate a vector
        IU sz = POPi;
        PUSH(mmu.tensor(sz))),
    CODE("matrix",                       ///< allocate a matrix
        IU w = POPi; IU h = POPi;
        PUSH(mmu.tensor(h, w))),
    CODE("tensor",                       ///< allocate a NHWC tensor
        IU c = POPi; IU w = POPi; IU h = POPi; IU n = POPi;
        PUSH(mmu.tensor(n, h, w, c))),
    CODE("vector{",                      ///< create a vector with literals
        IU sz = POPi;
        PUSH(mmu.tensor(sz));
        ten_off = 0; ten_lvl = 1),
    CODE("matrix{",                      ///< create a matrix with literals
        IU w = POPi; IU h = POPi;
        PUSH(mmu.tensor(h, w));
        ten_off = 0; ten_lvl = 1),
    CODE("view",   PUSH(mmu.view(top))), ///< create a view of a tensor
    CODE("copy",   PUSH(mmu.copy(top))), ///< create a hardcopy of a tensor
    ///@}
    ///@defgroup Tensor shape ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("flatten",                      ///< reshape as a vector (1-D array)
        Tensor &t = TTOS;
        t.reshape(t.numel)),
    CODE("reshape2",                     ///< reshape as matrix(h,w)
        IU w = POPi; IU h = POPi;
        TTOS.reshape(h, w)),
    CODE("reshape4",                     ///< reshape as Tensor(NHWC)
        IU c = POPi; IU w = POPi; IU h = POPi; IU n = POPi;
        TTOS.reshape(n, h, w, c)),
    ///@}
    ///@defgroup Tensor fill ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("={",                                    ///< (n -- ) or ( -- )
         ten_off = IS_OBJ(top) ? 0 : POPi;
         ten_lvl = IS_OBJ(top) ? 1 : 0),
    CODE("zeros", xop1(O_FILL, DU0)),             ///< fill tensor with 0s
    CODE("ones",  xop1(O_FILL, DU1)),             ///< fill tensor with 1s
    CODE("full",  DU d = POP(); xop1(O_FILL, d)), ///< fill tensor with a value
    CODE("eye",   xop1(O_IDEN)),                  ///< fill 1s in diag
    CODE("rand",  top = mmu.rand(top, UNIFORM)),  ///< uniform randomize a tensor or number
    CODE("randn", top = mmu.rand(top, NORMAL)),   ///< normal dist. randomize a tensor
    ///@}
    ///@defgrup Tensor slice and dice
    ///@{
    CODE("sum", if (TOS1T) PUSH(TTOS.sum())),
    CODE("avg", if (TOS1T) PUSH(TTOS.avg())),
    CODE("std", if (TOS1T) PUSH(TTOS.std())),
    CODE("{",   if (TOS1T && ten_lvl > 0) ++ten_lvl),
    CODE("}",   if (TOS1T && ten_lvl > 0) --ten_lvl),
    CODE("slice",
         IU y1 = POPi; IU y0 = POPi; IU x1 = POPi; IU x0 = POPi;
         if (TOS1T) {
             Tensor &t0 = TTOS;
             Tensor &t1 = mmu.slice(t0, x0, x1, y0, y1);
             PUSH(t1);
         }),
    ///@}
    ///@defgroup 1-tensor ops in-place (i.e. destructive, as in Forth)
    ///@{
    CODE("pow",       DU n = POP(); xop1(O_POW, n)),    ///< (A n -- A')
    CODE("exp",       xop1(O_EXP)),                     ///< (A -- A')
    CODE("ln",        xop1(O_LN)),                      ///< (A -- A')
    CODE("log",       xop1(O_LOG)),                     ///< (A -- A')
    ///@}
    ///@defgroup 1-tensor ops that create new tensor
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("inverse",   xop1x(O_INV)),      ///< (A -- A Ai')   matrix inversion (GaussJordan)
    CODE("det",       xop1x(O_DET)),      ///< (A -- A d)     matrix determinant
    CODE("lu",        xop1x(O_LU)),       ///< (A -- A A')    LU decomposition
    CODE("luinv",     xop1x(O_LUINV)),    ///< (A -- A A')    inverse the LU matrix
    CODE("upper",     xop1x(O_TRIU)),     ///< (A -- A A')    upper triangle
    CODE("lower",     xop1x(O_TRIL)),     ///< (A -- A A')    lower triangle
    CODE("transpose", xop1x(O_XPOS)),     ///< (A -- A At)    matrix transpose
    ///@}
    ///@defgroup 2-tensor matrix ops
    ///@{
    CODE("+=",        xop2(O_ADD, DROP)),
    CODE("-=",        xop2(O_SUB, DROP)),
    CODE("*=",        xop2(O_MUL, DROP)),
    CODE("/=",        xop2(O_DIV, DROP)),
    CODE("@=",        xop2(O_DOT, DROP)),
    CODE("matmul",    xop2(O_DOT, KEEP)), ///< (A B -- A B C) matrix multiply
    CODE("matdiv",                        ///< (A B -- A B C) matrix divide
        if (TOS2T) return;
        Tensor &A = TNOS; Tensor &B = TTOS;
        Tensor &C = _tdiv(A, B);
        if (C != B) PUSH(C)),
    CODE("solve",     xop2(O_SOLV,KEEP)), ///< (B A -- B A X) solve linear equations AX = B
    CODE("gemm",      _gemm()),           ///< (a b A B C -- a b A B C') GEMM (C updated)
    ///@}
    ///@defgroup Tensor persistance
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("bin",       PUSH(FAM_RAW)),     ///< raw/binary file
    CODE("w/o",       PUSH(FAM_WO)),      ///< write only file
    CODE("r/w",       PUSH(FAM_RW)),      ///< read-write file
    CODE("save",      _save()),           ///< save tensor
    };
    const Code ext[] = {                  ///< extended (overload) words
    ///@defgroup redefined tensor ops
    ///@{
    CODE("dolit",
        DU v = mmu.rd(IP); IP += sizeof(DU);
        if (IS_OBJ(v)) mmu.ref_inc(v);
        PUSH(v)),
    CODE("abs", xop1(O_ABS)),
    CODE(".",   tprint()),
    CODE("+",   xop2(O_ADD, KEEP)),
    CODE("-",   xop2(O_SUB, KEEP)),
    CODE("*",   xop2(O_MUL, KEEP)),
    CODE("/",   xop2(O_DIV, KEEP)),
    CODE("@",
        if (IS_OBJ(top)) xop2(O_DOT, KEEP);   ///< matrix @ product
        else {
            DU v = mmu.rd(POPi);
            if (IS_OBJ(v)) mmu.ref_inc(v);
            PUSH(v);
        }),
    CODE("+!",
        IU w = POPi; DU v = mmu.rd(w);
        if (IS_OBJ(v)) NA("obj +!");
        else mmu.wd(w, v + POP())),
    CODE("max",
        if (IS_OBJ(top)) PUSH(TTOS.max());
        else { DU n=ss.pop(); top = (top>n) ? top : n; }),
    CODE("min",
        if (IS_OBJ(top)) PUSH(TTOS.min());
        else { DU n=ss.pop(); top = (top<n) ? top : n; }),
    CODE("negate",
        if (IS_OBJ(top)) xop1(O_SCALE, -DU1);
        else top *= -DU1),
    ///@}
    CODE("boot", mmu.clear(FIND("gemm") + 1))
    };
    ForthVM::init();

    mmu.append(prim, sizeof(prim)/sizeof(Code)); /// * append tensor words
    mmu.merge(ext,   sizeof(ext)/sizeof(Code));  /// * overload existed words
    
    VLOG1("TensorVM::init ok\n");
};
///
/// override with tensor handler
///
__GPU__ int
TensorVM::number(char *str) {
    char *p;
    DU n = (STRCHR(idiom, '.'))
        ? STRTOF(idiom, &p)
        : STRTOL(idiom, &p, radix);
    if (*p != '\0') return 0;
    SCALAR(n);                           /// * mask out object bit
    if (compile) {                       /// * add literal when in compile mode
        VLOG2("%d| %f\n", vid, n);
        add_w(DOLIT);                    ///> dovar (+parameter field)
        add_du(n);                       ///> store literal
    }
    else if (ten_lvl > 0) {              /// * append literal into tensor storage
        VLOG2("%d| T[%d]=%f\n", vid, ten_off, n);
        TTOS.data[ten_off++] = n;        /// * append to tensor.data
    }
    else {                               ///> or, add value onto data stack
        VLOG2("%d| ss.push(%f)=%08x\n", vid, n, DU2X(n));
        PUSH(n);
    }
    return 1;
}
#endif  // T4_ENABLE_OBJ
//==========================================================================
