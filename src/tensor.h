/**
 * @file
 * @brief tensorForth tensor class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENSOR_H_
#define TEN4_SRC_TENSOR_H_
#include <ostream>
#include "t4base.h"
#include "util.h"
//===============================================================================
/// tensorForth tensor class
/// @brief - Tensor at rank=4, row-major, F32 only storage
/// Note:
///    PyTorch.Tensor: size, dtype, type_id, stride, tensorstore
///
typedef enum {
    /// 2-operand ops
    O_ADD = 0,
    O_SUB,
    O_MUL,
    O_DIV,
    O_DOT,
    O_SOLV,
    /// 1-operand + a constant
    O_FILL,
    O_SCALE,
    O_POW,
    /// 1-operand arithmetic ops
    O_ABS,
    O_EXP,
    O_LOG,
    O_TANH,
    O_RELU,
    O_SIGM,
    /// 1-operand matrix ops
    O_IDEN,
    O_INV,
    O_LU,
    O_LUINV,
    O_DET,
    O_TRIU,
    O_TRIL,
    O_XPOS
} t4_ten_op;

typedef enum {
    L_NONE = 0,
    L_CONV,
    L_LINEAR,
    L_FLATTEN,
    L_RELU,
    L_TANH,
    L_SIGMOID,
    L_SOFTMAX,
    L_MAXPOOL,
    L_AVGPOOL,
    L_MINPOOL,
    L_DROPOUT
} t4_layer;

typedef enum {
    MM_NONE  = 0,
    MM_INC   = 1,
    MM_A_TXP = 2,
    MM_B_TXP = 4
} t4_mm_opt;

struct Tensor : public T4Base {
    U16      stride[4] = {1,1,1,1}; ///< stride=HWCN, for calc memory offset
    U16      shape[4]  = {1,1,1,1}; ///< shape=HWCN, matrix C=N=1, vector W=C=N=1
    t4_layer grad_fn   = L_NONE;    ///< grandiant funtion type
    Tensor   *grad[4];              ///< gradiant and jacobian tensors
    ///
    /// static ops
    /// Note:
    ///   1. resultant tensor as last parameter
    ///   2. return the resultant tensor
    ///
    static __GPU__  Tensor &mm(Tensor &A, Tensor &B, Tensor &O, t4_mm_opt opt=MM_NONE);
    static __GPU__  Tensor &gemm(Tensor &A, Tensor &B, Tensor &O, DU alpha, DU beta);
    static __GPU__  Tensor &matx(t4_ten_op op, Tensor &A, Tensor &B, Tensor &O);  ///> matrix-matrix element-wise ops (Hadamard)
    static __GPU__  Tensor &matx(t4_ten_op op, Tensor &A, DU v, Tensor &O);       ///> matrix-scalar element-wise ops
    static __GPU__  Tensor &copy(Tensor &A, Tensor &O);
    static __GPU__  Tensor &transpose(Tensor &A, Tensor &T);
    static __GPU__  Tensor &inverse(Tensor &A, Tensor &I);  /// GaussJordan (with Pivot)
    static __GPU__  Tensor &lu(Tensor &A);                  /// LU (no Pivot)
    static __GPU__  Tensor &lu_inverse(Tensor &LU);         /// inverse a pre-processed LU (no Pivot)
    static __GPU__  Tensor &plu(Tensor &A, Tensor &P, int *ns);/// LU with permutation vector
    ///
    /// class contructors
    ///
    __HOST__ Tensor()       : T4Base() {}
    __HOST__ Tensor(U32 sz) : T4Base(sz) {
        shape[0] = (U16)sz;
        WARN("vector[%d] allocated\n", numel);
    }
    __HOST__ Tensor(U16 h, U16 w) : T4Base(h, w) {
        shape[0] = h; shape[1] = w;
        WARN("matrix(%d,%d) allocated\n", shape[0], shape[1]);
    }
    __HOST__ Tensor(U16 n, U16 h, U16 w, U16 c) : T4Base(n, h, w, c) {
        shape[0] = h; shape[1] = w; shape[2] = n; shape[3] = c;
        WARN("tensor(%d,%d,%d,%d) allocated\n", shape[3], shape[0], shape[1], shape[2]);
    }
    __HOST__ ~Tensor() {
        switch (rank) {
        case 2: WARN("matrix(%d,%d) freed\n", shape[0], shape[1]); break;
        case 4: WARN("tensor(%d,%d,%d,%d) freed\n", shape[3], shape[0], shape[1], shape[2]); break;
        default: WARN("~Tensor error: rank=%d\n", rank);
        }
    }
    ///
    /// attributes
    ///
    __BOTH__ __INLINE__ U16  N() { return shape[3]; }
    __BOTH__ __INLINE__ U16  H() { return shape[0]; }
    __BOTH__ __INLINE__ U16  W() { return shape[1]; }
    __BOTH__ __INLINE__ U16  C() { return shape[2]; }
    __GPU__  __INLINE__ bool is_same_shape(Tensor &t) {
        return MEMCMP(shape, t.shape, sizeof(shape)) == 0;
    }
    ///
    /// tensor arithmetics
    ///
    __GPU__  DU     sum();
    __GPU__  DU     avg();
    __GPU__  DU     std();    // standard distribution
    __GPU__  DU     max();
    __GPU__  DU     min();
    __GPU__  DU     dot(Tensor &B);
    ///
    /// linear algebra methods
    ///
    __GPU__  DU     det();                    ///< matrix determinant
    __GPU__  Tensor &triu();                  ///< upper triangle
    __GPU__  Tensor &tril();                  ///< lower triangle
    ///
    /// tensor life-cycle ops
    ///
    __BOTH__ Tensor &reset(void *mptr, U32 sz, t4_obj tt=T4_TENSOR, t4_layer fn=L_NONE);
    __BOTH__ Tensor &reshape(U32 sz);
    __BOTH__ Tensor &reshape(U16 h, U16 w);
    __BOTH__ Tensor &reshape(U16 n, U16 h, U16 w, U16 c);
    __BOTH__ Tensor &reshape(U16 c1, U16 n, U16 h, U16 w, U16 c);
    
    __BOTH__ Tensor &identity();              ///< fill as an identity matrix
    __BOTH__ Tensor &map(t4_ten_op op, DU v=DU0); ///< element-wise absolute
    __BOTH__ Tensor &fill(DU v) { return this->map(O_FILL, v); }
    __HOST__ void   copy_to_host(void* dst) { cudaMemcpy(dst, data, numel, cudaMemcpyDeviceToHost); }
    ///
    /// IO
    ///
    __BOTH__ void to_s(std::ostream &fout);
    ///
    /// TODO: tensor arithmetics
    ///
    __GPU__ __INLINE__ Tensor &operator+=(Tensor &t){ matx(O_ADD, *this, t, *this); return *this; }
    __GPU__ __INLINE__ Tensor &operator-=(Tensor &t){ matx(O_SUB, *this, t, *this); return *this; }
    __GPU__ __INLINE__ Tensor &operator*=(Tensor &t){ matx(O_MUL, *this, t, *this); return *this; }
    ///
    /// TODO: tensor logical ops
    ///
    __GPU__ __INLINE__ bool   operator<(Tensor &t)  { return 0; }
    __GPU__ __INLINE__ bool   operator>(Tensor &t)  { return 0; }
    __GPU__ __INLINE__ bool   operator<=(Tensor &t) { return 0; }
    __GPU__ __INLINE__ bool   operator>=(Tensor &t) { return 0; }
    __GPU__ __INLINE__ bool   operator==(Tensor &t) { return 0; }
};
#endif // TEN4_SRC_TENSOR_H_
