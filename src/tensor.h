/**
 * @file
 * @brief tensorForth tensor class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENSOR_H_
#define TEN4_SRC_TENSOR_H_
#include "ten4_types.h"
#include "vector.h"
/**
  TODO: Matrix product of two Tensors.
  The behavior depends on the dimensionality of the Tensors as follows:
  - If both Tensors are 1-dimensional, the dot product (scalar) is returned.
  - If both arguments are 2-dimensional, the matrix-matrix product is returned.
  - If the first argument is 1-dimensional and the second argument is 2-dimensional,
    a 1 is prepended to its dimension for the purpose of the matrix multiply.
    After the matrix multiply, the prepended dimension is removed.
  - If the first argument is 2-dimensional and the second argument is 1-dimensional,
    the matrix-vector product is returned.
  - If both arguments are at least 1-dimensional and at least one argument is
    N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
    argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
    batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
    1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
    The non-matrix (i.e. batch) dimensions are broadcasted (and thus
    must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
    and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
//===============================================================================
/// tensorForth tensor class
/// @brief - Tensor at rank=4, row-major, F32 only storage
/// Note:
///    PyTorch.Tensor: size, dtype, type_id, stride, tensorstore
///
struct Tensor : public Managed {
    U64              size;      ///< number of contiguous bytes
    U32              dsize;     ///< size of data element, F32 for now, TODO: others
    U32              rank;      ///< rank of tensor 2:matrix, 4:NHWC tensor
    U16              stride[4]; ///< strides to calculate memory offset
    U16              shape[4];  ///< Tensor4 (HWNC), matrix N=0, C=0
    void             *data = 0; ///< managed memory block pointer
    union {
        DU           f;         ///< float storage
        struct {
            U32 t  : 1;         ///< tensor rank >= 1
            U32 idx: 31;        ///< tensor pool index (2^31 slots)
        };
    };
    __BOTH__ Tensor(DU f0): f(f0)  { t = 0; }
    __HOST__ Tensor();
    __HOST__ Tensor(U16 n, U16 h, U16 w, U16 c);
    __HOST__ Tensor(U16 h, U16 w);
    __HOST__ Tensor(U64 sz);
    __HOST__ ~Tensor();
    
    __BOTH__ U16 N() { return shape[2]; }
    __BOTH__ U16 H() { return shape[0]; }
    __BOTH__ U16 W() { return shape[1]; }
    __BOTH__ U16 C() { return shape[3]; }
    ///
    /// tensor reshape and assignment
    ///
    __BOTH__ Tensor &reset(void *mptr, U64 sz);
    __BOTH__ Tensor &reshape(U16 h, U16 w);
    __BOTH__ Tensor &reshape(U16 n, U16 h, U16 w, U16 c);
    __HOST__ Tensor &fill(U8 v=0);
    __HOST__ Tensor &random(int seed=0);
    __HOST__ void   copy_to(void* dst) { cudaMemcpy(dst, data, size, cudaMemcpyDeviceToHost); }
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
    ///
    /// GEMM ops
    ///
    __GPU__ Tensor &gemm(Tensor &A, Tensor &B, Tensor &C);
    __GPU__ Tensor &grad(Tensor &A, Tensor &B, Tensor &C);
};
#endif // TEN4_SRC_TENSOR_H_
