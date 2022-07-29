/** -*- c++ -*-
 * @File
 * @brief - eForth Vritual Machine implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "tenvm.h"

#if T4_ENABLE_OBJ
__GPU__ void
TensorVM::tprint(DU d) {
    if (IS_OBJ(d)) { fout << d; mmu.mark_free(d); }
    else fout << " " << d;                  /// eForth has a space prefix
}
__GPU__ void TensorVM::add_to_tensor(DU n) {
    DU *d = (DU*)mmu.du2ten(top).data;
    d[ten_off++] = n;
}
///
/// tensor methods
///
__GPU__ void
TensorVM::texp() {
    if (!IS_TEN(top)) {                      /// * scalar
        top = EXP(top); NO_OBJ(top);         /// * mask off object-bit if any
        return;
    }
    Tensor &A = mmu.du2ten(top);
    Tensor &B = mmu.copy(A);
    DU *d = (DU*)B.data;
    for (int i=0; i < B.size; i++, d++) {    ///> TODO: CDP
        *d = EXP(*d);
    }
    PUSH(B);
}
__GPU__ void
TensorVM::tadd(bool sub) {
    bool s0 = !IS_TEN(top), s1 = !IS_TEN(ss[-1]);
    
    if (s0 && s1) {
        top = sub ? ss.pop() - top : ss.pop() + top;
        return;
    }
    if (s0 || s1) { ERROR("dim?"); return; }  ///> TODO: broadcast
    
    Tensor &A = mmu.du2ten(ss[-1]);           ///> tensor +- tensor
    Tensor &B = mmu.du2ten(top);
    U16 h = A.H(), w = A.W();
    if (h == B.H() && w == B.W()) {
        Tensor &C = mmu.tensor(h, w);
        Tensor::add(A, B, C, sub);
        PUSH(C);
        return;
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
TensorVM::tmul() {                                    ///< tensor multiplication
    bool s0 = !IS_TEN(top), s1 = !IS_TEN(ss[-1]);     /// * scalar check
    if (s0 && s1) { top *= ss.pop(); return; }        /// * scalar * scalar
    
    Tensor &A = mmu.du2ten(s1 ? top : ss[-1]);
    if (s0 || s1) {                                   /// * tensor * scalar
        Tensor &C = mmu.copy(A);                      /// * hard copy A tensor
        DU     k  = s0 ? top : ss[-1];
        VLOG2("T%d=%p * %f => A'=%p\n", A.rank, &A, k, &C);
        mmu.ten2du(C.scale(k));                       /// * resultant tensor on TOS
        PUSH(C);
        return;
    }
    Tensor &B = mmu.du2ten(top);                      /// tensor * tensor
    U16 m = A.H(), ka = A.W(), kb = B.H(), n = B.W();
    VLOG2("A[%d,%d]=%p x B[%d,%d]=%p ", m, ka, &A, kb, n, &B);
    if (A.rank==1 && B.rank==1 && A.size==B.size) {   /// * vector x vector
        PUSH(A.dot(B));                               /// * dot product on TOS
        VLOG2(" => %f\n", top);
    }
    else if (ka == kb) {                              /// * tensor x tensor
        Tensor &C = mmu.tensor(m, n);
        Tensor::mm(A, B, C);
        PUSH(mmu.ten2du(C));                          /// * resultant tensor on TOS
        VLOG2("=> C[%d,%d]=%p\n", C.H(), C.W(), &C);
    }
    else ERROR("dim?");
}
__GPU__ void
TensorVM::tdiv() {                                     ///< tensor division
    if (!IS_TEN(ss[-1])) {
        top = ss.pop() / top;                          /// * scalar / scalar
        NO_OBJ(top);
        return;
    }
    Tensor &A = mmu.du2ten(ss[-1]);
    if (!IS_TEN(top)) {                               /// * tensor / scalar
        Tensor &C = mmu.copy(A);                      /// * hard copy A tensor
        VLOG2("A[%d,%d]=%p / %f => A'=%p\n", A.H(), A.W(), &A, top, &C);
        top = mmu.ten2du(C.scale(1.0/top));           /// * resultant tensor on TOS
        return;
    }
    /// TODO: tensor * inverse(tensor)
}
///
/// matrix inversion GauseJordan (with Pivot)
///
__GPU__ void
TensorVM::tinv() {
    if (!IS_TEN(top)) { ERROR("tensor?"); return; }
    Tensor &A = mmu.du2ten(top);
    Tensor &I = mmu.tensor(A.H(), A.W()).identity();
    Tensor &C = mmu.copy(A);
    Tensor::inverse(C, I);
    mmu.free(C);
    PUSH(I);
}
///
/// LU conversion (no Pivot)
///
__GPU__ void
TensorVM::tlu() {
    if (!IS_TEN(top)) { ERROR("tensor?"); return; }
    Tensor &A  = mmu.du2ten(top);
    Tensor &LU = mmu.copy(A);             /// * hardcopy original matrix
    Tensor::lu(LU);                       /// * decompose A to LU
    PUSH(LU);
}
///
/// inverse of a LU matrix (no Pivot)
///
__GPU__ void
TensorVM::tluinv() {
    tlu();
    Tensor &LU = mmu.du2ten(top);
    Tensor::inverse(LU);
}
///
/// matrix determinant
///
__GPU__ void
TensorVM::tdet() {
    if (!IS_TEN(top)) { ERROR("tensor?"); return; }
    Tensor &A  = mmu.du2ten(top);
    Tensor &LU = mmu.copy(A);             /// * hardcopy original matrix
    Tensor::lu(LU);                       /// * decompose A to LU
    PUSH(LU.det());
}
__GPU__ void
TensorVM::ttrans() {
    if (!IS_TEN(top)) { ERROR("tensor?"); return; }
    Tensor &A = mmu.du2ten(top);
    U16 h = A.H(), w = A.W();
    Tensor &B = mmu.tensor(w, h);
    VLOG2("A[%d,%d]=%p => B[%d,%d]=%p", h, w, &A, B.H(), B.W(), &B);
    Tensor::transpose(A, B);
    PUSH(B);
}
__GPU__ void
TensorVM::gemm() {                       ///< blas GEMM
    Tensor &C = mmu.du2ten(top);
    Tensor &B = mmu.du2ten(ss[-1]);
    Tensor &A = mmu.du2ten(ss[-2]);
    DU     b  = ss[-3];
    DU     a  = ss[-4];
    U16 m = A.H(), k = A.W(), n = B.W();
    if (k == B.H() && m == C.H() && n == C.W()) {
        Tensor &D = mmu.copy(C);         /// * hard copy C tensor
        Tensor::gemm(A, B, D, a, b);
        PUSH(D);
    }
    else ERROR("dim?");
}
///
/// Tensor specific dictionary constructor
///
__GPU__ void
TensorVM::init_t() {
    const Code prim[] = {       /// singleton, build once only
    ///@defgroup Tensor creation ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("vector",                        ///< allocate a vector
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
    CODE("copy",    PUSH(mmu.copy(top))),
    ///@}
    ///@defgroup Tensor shape ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("flatten",                      ///< reshape as a vector (1-D array)
        Tensor &t = mmu.du2ten(top);
        t.reshape(t.size)),
    CODE("reshape2",                     ///< reshape as matrix(h,w)
        IU w = POPi; IU h = POPi;
        mmu.du2ten(top).reshape(h, w)),
    CODE("reshape4",                     ///< reshape as Tensor(NHWC)
        IU c = POPi; IU w = POPi; IU h = POPi; IU n = POPi;
        mmu.du2ten(top).reshape(n, h, w, c)),
    ///@}
    ///@defgroup Tensor fill ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("={",                          ///< (n -- ) or ( -- )
         ten_off = IS_TEN(top) ? 0 : POPi;
         ten_lvl = IS_TEN(top) ? 1 : 0),
    CODE("zeros", if (IS_TEN(top)) mmu.du2ten(top).fill(0)),
    CODE("ones",  if (IS_TEN(top)) mmu.du2ten(top).fill(1)),
    CODE("full",  if (IS_TEN(ss[-1])) { DU d = POP(); mmu.du2ten(top).fill(d); }),
    CODE("eye",   if (IS_TEN(top)) mmu.du2ten(top).identity()),
    CODE("rand",  top = mmu.rand(top, UNIFORM)),  ///< uniform randomize a tensor or number
    CODE("randn", top = mmu.rand(top, NORMAL)),   ///< normal dist. randomize a tensor
    ///@defgrup Tensor slice and dice
    ///@{
    CODE("sum",
        if (IS_TEN(top)) {
            DU d =  mmu.du2ten(top).sum();
            PUSH(d);
        }),
    CODE("{",   if (IS_TEN(top) && ten_lvl > 0) ++ten_lvl),
    CODE("}",   if (IS_TEN(top) && ten_lvl > 0) --ten_lvl),
    CODE("slice",
         IU y1 = POPi; IU y0 = POPi; IU x1 = POPi; IU x0 = POPi;
         if (IS_TEN(top)) {
             Tensor &t0 = mmu.du2ten(top);
             Tensor &t1 = mmu.slice(t0, x0, x1, y0, y1);
             PUSH(t1);
         }),
    ///@}
    ///@}
    ///@defgroup Tensor matrix ops
    ///@brief - stick to PyTorch naming when possible
    ///@{
    CODE("exp",       texp()),     ///< (A -- A A')    matrix exponential
    CODE("inverse",   tinv()),     ///< (A -- A Ai')   matrix inversion (GaussJordan)
    CODE("lu",        tlu()),      ///< (A -- A A')    LU decomposition
    CODE("luinv",     tluinv()),   ///< (A -- A A')    matrix inverse via LU
    CODE("det",       tdet()),     ///< (A -- A d)     matrix determinant
    CODE("upper", if (!IS_TEN(top)) return;
         Tensor &t0 = mmu.du2ten(top);
         Tensor &t1 = mmu.copy(t0);
         t1.triu();
         PUSH(t1)),
    CODE("lower", if (!IS_TEN(top)) return;
         Tensor &t0 = mmu.du2ten(top);
         Tensor &t1 = mmu.copy(t0);
         t1.tril();
         PUSH(t1)),
    CODE("transpose", ttrans()),   ///< (A -- A At)    matrix transpose
    CODE("matmul",    tmul()),     ///< (A B -- A B C) matrix multiplication
    CODE("gemm",      gemm()),     ///< (a b A B C -- a b A B C') GEMM (C updated)
    ///@}
    };
    const Code old[] = {
    ///@defgroup redefined tensor ops
    ///@{
    CODE(".",   tprint(POP())),
    CODE("+",   tadd()),
    CODE("*",   tmul()),
    CODE("-",   tadd(true)),
    CODE("/",   tdiv()),
    ///@}
    CODE("boot", mmu.clear(FIND("gemm") + 1))
    };
    ForthVM::init_f();

    mmu.append(prim, sizeof(prim)/sizeof(Code));    /// * append tensor words
    mmu.merge(old,  sizeof(old)/sizeof(Code));      /// * update existed words
    mmu.status();

    status = VM_RUN;
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
    NO_OBJ(n);                           /// * mask out object bit
    if (compile) {                       /// * add literal when in compile mode
        VLOG2("%f\n", n);
        add_w(DOLIT);                    ///> dovar (+parameter field)
        add_du(n);                       ///> store literal
    }
    else if (ten_lvl > 0) {              /// * append literal into tensor storage
        VLOG2("T[%d]=%f\n", ten_off, n);
        add_to_tensor(n);
    }
    else {                               ///> or, add value onto data stack
        VLOG2("ss.push(%08x)\n", *(U32*)&n);
        PUSH(n);
    }
}
#endif  // T4_ENABLE_OBJ
//=======================================================================================
