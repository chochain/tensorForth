/** -*- c++ -*-
 * @File
 * @brief - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tenvm.h"

#if T4_ENABLE_OBJ
///
/// Tensor print to host
///
__GPU__ void
TensorVM::tprint(DU d) {
    if (IS_OBJ(d)) { fout << d; mmu.mark_free(d); }
    else fout << " " << d;                  /// eForth has a space prefix
}
///
/// Tensor-self ops
///
__GPU__ void
TensorVM::xop1(t4_ten_op op, DU v) {
    if (!IS_OBJ(top)) {                     /// * scalar value
        switch (op) {
        case O_ABS:  top = ABS(top);          break;
        case O_EXP:  top = EXP(top);          break;
        case O_TANH: top = tanh(top);         break;
        case O_RELU: top = top > v ? top : v; break;
        }
        SCALAR(top);
        return;
    }
    Tensor &A = TTOS;
    if (!A.is_tensor()) { ERROR("tensor?"); return; }
    
    switch (op) {
    /// ops => update in-place
    case O_FILL:
    case O_SCALE: A.map(op, v);   break;
    case O_ABS:
    case O_EXP:   
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
    
    Tensor &t  = mmu.copy(A);                 /// * hardcopy original matrix
    bool   tos = true;
    switch (op) {
    case O_INV:
        mmu.free(t);                          /// * not needed
        PUSH(_tinv(A));                       /// * _tinv create its own temp
        tos = false;              break;
    case O_DET: {                             /// * TODO: use PLU
        Tensor::lu(t);                        /// * decompose A to LU
        PUSH(t.det());                        /// * return determinant on TOS
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
        tos = false;
    }
    if (tos) PUSH(t);
}
__GPU__ void
TensorVM::xop2(t4_ten_op op, t4_drop_opt x) {
    bool s0 = !IS_OBJ(top), s1 = !IS_OBJ(ss[-1]); /// * scalar flags
    if (s0 && s1) return _ss_op(op);              /// * op(scalar, scalar)
    if (s0 || s1) return _ts_op(op, x, s1);       /// * op(tensor, scalar)
    _tt_op(op, x);                                /// * op(tensor, tensor)
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
    case O_DIV: top = DIV(ss.pop(), top); SCALAR(top); break;
    }
}
__GPU__ void
TensorVM::_ts_op(t4_ten_op op, t4_drop_opt x, bool swap) { ///< tensor-scalar ops
    auto drop = [this](Tensor &t) { POP(); mmu.free(t); };

    Tensor &A = swap ? TTOS : TNOS;
    DU     v  = swap ? ss[-1] : top;
    Tensor &C = mmu.tensor(A.H(), A.W());
    if (swap && (op==O_DIV || op==O_SUB)) {   /// * op(scaler, tensor)
        Tensor &B = mmu.tensor(A.numel);      /// * working tensor
        B.map(O_FILL, v);                     /// * broadcast
        Tensor::mat(op, B, A, C);             /// * Hadamard ops
        mmu.free(B);                          /// * free working tensor
    }
    else Tensor::mat(op, A, v, C);            /// * broadcast_op(tensor, scalar)

    static const char *opn[] = { "+", "-", "*", "/", "@", "x" };
    VLOG1("A[%d,%d] %s %f => C[%d,%d]\n", A.H(), A.W(), opn[op], v, C.H(), C.W());
    if (x==DROP) { drop(A); POP(); }          /// TODO: in-place
    PUSH(C);
}
///
/// op(tensor, tensor)
///
__GPU__ void
TensorVM::_tt_op(t4_ten_op op, t4_drop_opt x) {///< tensor-tensor ops
    Tensor &A = TNOS, &B = TTOS;
    ///
    /// broadcast_op(tensor, tensor)
    ///
    Tensor *C = NULL;
    bool   tt = true;                        ///< tensor-tensor flag
    switch (op) {                            ///> op(tensor, tensor)
    case O_DOT:  C = _tdot(A, B, &tt); break;
    case O_SOLV: C = _solv(A, B);      break;
    default:                                 ///> op(tensor, tensor) Hadamard
        if (A.is_same_shape(B)) {            /// * match sizes
            C = (A.rank==1 && B.rank==1)
                ? &mmu.tensor(A.H())
                : &mmu.tensor(A.H(), A.W());
            Tensor::mat(op, A, B, *C);       /// * Hadamard ops
            if (A.rank==1 && B.rank==1) C->reshape(C->numel);
        }
    }
    if (tt && C) {
        static const char *opn[] = { "+", "-", "*", "/", "@", "x" };
        VLOG1("TensorVM# A[%d,%d] %s B[%d,%d] => C[%d,%d]\n",
              A.H(), A.W(), opn[op], B.H(), B.W(), C->H(), C->W());
        if (x==DROP) { POP(); mmu.free(B); POP(); mmu.free(A); }
        PUSH(*C);
    }
    else if (tt) ERROR("dim?");
}
__GPU__ Tensor&
TensorVM::_tinv(Tensor &A) {                 ///< matrix inverse
    Tensor &I = mmu.tensor(A.H(), A.W()).identity();
    Tensor &X = mmu.copy(A);
    Tensor::inverse(X, I);
    mmu.free(X);                             /// * release temp 
    return I;
}
__GPU__ Tensor*
TensorVM::_tdiv(Tensor &A, Tensor &B) {      ///< tensor division
    U16 m = A.H(), ka = A.W(), kb = B.H(), n = B.W();
    if (kb != n || ka != kb) return NULL;    /// * B square?

    Tensor &I = _tinv(B);
    Tensor &C = mmu.tensor(m, n);
    Tensor::mm(A, I, C);                     /// A * B^-1
    mmu.free(I);
    return &C;
}
__GPU__ Tensor*
TensorVM::_tdot(Tensor &A, Tensor &B, bool *tt) {///< tensor dot product
    Tensor *C = NULL;
    if (B.rank==1 &&                         ///> dot(vector, vector)
        A.rank==1 && A.numel==B.numel) {
        DU v = A.dot(B);
        PUSH(v);
        *tt = false;
        VLOG1("A[%d] @ B[%d] => %f\n", A.H(), B.H(), v);
    }
    else if (B.rank==1 && A.W()==B.numel) {  ///> inner(tensor, vector)
        C = &mmu.tensor(A.H());
        Tensor::mm(A, B, *C);
    }
    else if (A.W()==B.H()) {                 /// * tensor @ tensor
        C = &mmu.tensor(A.H(), B.W());
        Tensor::mm(A, B, *C);
    }
    return C;
}
__GPU__ Tensor*
TensorVM::_solv(Tensor &B, Tensor &A) {      /// Note: A B flipped [3,3]x[3,1]
    printf("solv[%d,%d]x[%d,%d]\n", A.H(), A.W(), B.H(), B.W());
    U16 m = A.H(), k = A.W(), n = B.H();
    if (B.rank!=1 || m!=k || k!=n) return NULL;
    
    Tensor &I = _tinv(A);
    Tensor &C = mmu.tensor(k);               /// resultant vector
    Tensor::mm(I, B, C);                     /// C = A^-1 x B
    mmu.free(I);
    return &C;
}
__GPU__ void
TensorVM::_gemm() {                          ///< blas GEMM
    if (!TOS3T) { ERROR("tensors?"); return; }
    
    Tensor &C = TTOS, &B = TNOS, &A = (Tensor&)mmu.du2obj(ss[-2]);
    DU     b  = ss[-3];
    DU     a  = ss[-4];
    U16    m  = A.H(), k = A.W(), n = B.W();
    if (k == B.H() && m == C.H() && n == C.W()) {
        Tensor &D = mmu.copy(C);             /// * hard copy C tensor
        Tensor::gemm(A, B, D, a, b);
        PUSH(D);
    }
    else ERROR("dim?");
}
///
/// Tensor specific dictionary constructor
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
    CODE("exp",       xop1(O_EXP)),        ///< (A -- A')
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
        Tensor *C = _tdiv(A, B);
        PUSH(*C)),
    CODE("solve",     xop2(O_SOLV,KEEP)), ///< (B A -- B A X) solve linear equations AX = B
    CODE("gemm",      _gemm()),           ///< (a b A B C -- a b A B C') GEMM (C updated)
    ///@}
    };
    const Code ext[] = {                  ///< extended (overload) words
    ///@defgroup redefined tensor ops
    ///@{
    CODE("abs",      xop1(O_ABS)),
    CODE(".",        tprint(POP())),
    CODE("+",        xop2(O_ADD, KEEP)),
    CODE("-",        xop2(O_SUB, KEEP)),
    CODE("*",        xop2(O_MUL, KEEP)),
    CODE("/",        xop2(O_DIV, KEEP)),
    CODE("@",
         if (IS_OBJ(top)) xop2(O_DOT, KEEP);
         else {
             IU w = POPi; PUSH(mmu.rd((IU)w));
         }),
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
    mmu.status();
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
        VLOG2("%f\n", n);
        add_w(DOLIT);                    ///> dovar (+parameter field)
        add_du(n);                       ///> store literal
    }
    else if (ten_lvl > 0) {              /// * append literal into tensor storage
        VLOG2("T[%d]=%f\n", ten_off, n);
        TTOS.data[ten_off++] = n;        /// * append to tensor.data
    }
    else {                               ///> or, add value onto data stack
        VLOG2("ss.push(%08x)\n", *(U32*)&n);
        PUSH(n);
    }
    return 1;
}
#endif  // T4_ENABLE_OBJ
//=======================================================================================
