/**
 * @file
 * @brief tensorForth tensor class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENSOR_H_
#define TEN4_SRC_TENSOR_H_
#include <ostream>
#include "ten4_types.h"
#include "util.h"
//===============================================================================
/// tensorForth tensor class
/// @brief - Tensor at rank=4, row-major, F32 only storage
/// Note:
///    PyTorch.Tensor: size, dtype, type_id, stride, tensorstore
///
typedef enum {
    ADD = 0,
    SUB,
    MUL,
    DIV,
    ABS,
    EXP,
    TANH,
    RELU
} mat_op;

#define T4_TENSOR_VIEW  1
struct Tensor : public Managed {
    U32              size;      ///< number of data elements, TODO: more than 4G elements
    U16              dsize;     ///< size of data element, F32 for now, TODO: others
    U16              rank;      ///< rank of tensor 2:matrix, 4:NHWC tensor
    U16              stride[4]; ///< strides to calculate memory offset
    U16              shape[4];  ///< shape=HWCN, matrix C=N=1, vector W=C=N=1
    U32              attr = 0;  ///< tensor attributes (a view)
    union {
        U8           *data = 0; ///< managed memory block pointer
        DU           f;         ///< float storage
        struct {
            U32 t  : 1;         ///< tensor rank >= 1
            U32 idx: 31;        ///< tensor pool index (2^31 slots)
        };
    };
    ///
    /// static ops
    /// Note:
    ///   1. resultant tensor as last parameter
    ///   2. return the resultant tensor
    ///
    static __BOTH__ Tensor &gemm(Tensor &A, Tensor &B, Tensor &C, DU alpha, DU beta);
    static __BOTH__ Tensor &grad(Tensor &A, Tensor &B, Tensor &C);
    static __BOTH__ Tensor &mm(Tensor &A, Tensor &B, Tensor &C) { return gemm(A, B, C, 1.0, 0.0); }
    static __BOTH__ Tensor &mat(mat_op op, Tensor &A, Tensor &B, Tensor &C);  ///> matrix-matrix element-wise ops (Hadamard)
    static __BOTH__ Tensor &mat(mat_op op, Tensor &A, DU v, Tensor &C);       ///> matrix-scalar element-wise ops
    static __BOTH__ Tensor &copy(Tensor &A, Tensor &C);
    static __BOTH__ Tensor &transpose(Tensor &A, Tensor &T);
    static __BOTH__ Tensor &inverse(Tensor &A, Tensor &I);  /// GaussJordan (with Pivot)
    static __BOTH__ Tensor &inverse(Tensor &LU);            /// from LU (no Pivot)
    static __BOTH__ Tensor &lu(Tensor &A);                  /// LU (no Pivot)
    static __BOTH__ Tensor &plu(Tensor &A, Tensor &P);      /// LU with permutation vector
    ///
    /// class contructors
    ///
    __HOST__ Tensor();
    __HOST__ Tensor(U16 n, U16 h, U16 w, U16 c);
    __HOST__ Tensor(U16 h, U16 w);
    __HOST__ Tensor(U32 sz);
    __HOST__ ~Tensor();
    __HOST__ Tensor(DU f0): f(f0)  { t = 0; }
    ///
    /// attributes
    ///
    __BOTH__ __INLINE__ U16  N()       { return shape[3]; }
    __BOTH__ __INLINE__ U16  H()       { return shape[0]; }
    __BOTH__ __INLINE__ U16  W()       { return shape[1]; }
    __BOTH__ __INLINE__ U16  C()       { return shape[2]; }
    __BOTH__ __INLINE__ bool is_view() { return attr & T4_TENSOR_VIEW; }
    ///
    /// tensor arithmetics
    ///
    __BOTH__ DU     sum();
    __BOTH__ DU     dot(Tensor &B);
    __BOTH__ Tensor &math(mat_op op);         ///< element-wise absolute
    ///
    /// linear algebra methods
    ///
    __BOTH__ DU     det();                    ///< matrix determinant
    __BOTH__ Tensor &triu();                  ///< upper triangle
    __BOTH__ Tensor &tril();                  ///< lower triangle
    __BOTH__ Tensor &scale(DU v);             ///< element-wise linear scale
    ///
    /// tensor life-cycle ops
    ///
    __BOTH__ Tensor &set_as_view(bool set=true);
    __BOTH__ Tensor &reset(void *mptr, U32 sz);
    __BOTH__ Tensor &reshape(U32 sz);
    __BOTH__ Tensor &reshape(U16 h, U16 w);
    __BOTH__ Tensor &reshape(U16 n, U16 h, U16 w, U16 c);
    __BOTH__ Tensor &fill(DU v);
    __BOTH__ Tensor &identity();              ///< fill as an identity matrix
    __HOST__ void   copy_to_host(void* dst) { cudaMemcpy(dst, data, size, cudaMemcpyDeviceToHost); }
    ///
    /// IO
    ///
    __BOTH__ void to_s(std::ostream &fout);
    ///
    /// assignment
    ///
    __BOTH__ __INLINE__ Tensor &operator=(DU f0) { f = f0; t = 0; return *this; }
    ///
    /// tensor arithmetics
    ///
    __BOTH__ __INLINE__ Tensor &operator+=(Tensor &t){ f += t.f; return *this; }
    __BOTH__ __INLINE__ Tensor &operator-=(Tensor &t){ f -= t.f; return *this; }
    __BOTH__ __INLINE__ F32    operator+(Tensor &t)  { return f + t.f; }
    __BOTH__ __INLINE__ F32    operator-(Tensor &t)  { return f - t.f; }
    __BOTH__ __INLINE__ F32    operator*(Tensor &t)  { return f * t.f; }
    __BOTH__ __INLINE__ F32    operator/(Tensor &t)  { return f / t.f; }
    __BOTH__ __INLINE__ F32    operator%(Tensor &t)  { return fmod(f, t.f); }
    ///
    /// tensor logical ops
    ///
    __BOTH__ __INLINE__ bool   operator<(Tensor &t)  { return (f - t.f) <  -DU_EPS; }
    __BOTH__ __INLINE__ bool   operator>(Tensor &t)  { return (f - t.f) >   DU_EPS; }
    __BOTH__ __INLINE__ bool   operator<=(Tensor &t) { return (f - t.f) <= -DU_EPS; }
    __BOTH__ __INLINE__ bool   operator>=(Tensor &t) { return (f - t.f) >=  DU_EPS; }
    __BOTH__ __INLINE__ bool   operator==(Tensor &t) { return fabs(f - t.f) <  DU_EPS; }
    __BOTH__ __INLINE__ bool   operator!=(Tensor &t) { return fabs(f - t.f) >= DU_EPS; }
    ///
    /// float arithmetics
    ///
    __BOTH__ __INLINE__ Tensor &operator+=(F32 f0)   { f += f0; t = 0; return *this; }
    __BOTH__ __INLINE__ Tensor &operator-=(F32 f0)   { f -= f0; t = 0; return *this; }
    __BOTH__ __INLINE__ Tensor &operator*=(F32 f0)   { f *= f0; t = 0; return *this; }
    __BOTH__ __INLINE__ Tensor &operator/=(F32 f0)   { f /= f0; t = 0; return *this; }
    ///
    /// float logical ops
    ///
    __BOTH__ __INLINE__ bool   operator<(F32 f0)     { return (f - f0) <  -DU_EPS; }
    __BOTH__ __INLINE__ bool   operator>(F32 f0)     { return (f - f0) >   DU_EPS; }
    __BOTH__ __INLINE__ bool   operator>=(F32 f0)    { return (f - f0) >=  DU_EPS; }
    __BOTH__ __INLINE__ bool   operator==(F32 f0)    { return fabs(f - f0)  <  DU_EPS; }
    __BOTH__ __INLINE__ bool   operator!=(F32 f0)    { return fabs(f - f0)  >= DU_EPS; }
};
#endif // TEN4_SRC_TENSOR_H_
