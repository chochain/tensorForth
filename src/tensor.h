/**
 * @file
 * @brief tensorForth tensor class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_TENSOR_H_
#define TEN4_SRC_TENSOR_H_
#include "ten4_types.h"
///
///@name Data conversion
///@{
#define INT(t)    (static_cast<int>(t.f))       /**< cast float to int                       */
#define ABS(t)    (fabs(t.f))                   /**< absolute value                          */
///}
//===============================================================================
/// tensorForth tensor class
///@}
///@name tensorForth complex data object
///@{
#define __XNLINE__
typedef struct Tensor {
    union {
        F32 f;
        struct {
            U32 r: 1;       // tensor rank >= 1
            U32 p: 31;      // 2^31 slots
        };
    };
    __BOTH__ Tensor()      : f(DU0) { r = 0; }
    __BOTH__ Tensor(F32 f0): f(f0)  { r = 0; }
    __BOTH__ __XNLINE__ Tensor &operator=(S32 i)     { f = static_cast<F32>(i); return *this; }
    __BOTH__ __XNLINE__ Tensor &operator=(F32 f0)    { f = f0; r = 0; return *this; }
	///
    /// tensor arithmetics
    ///
    __BOTH__ __XNLINE__ Tensor &operator+=(Tensor &t){ f += t.f; return *this; }
    __BOTH__ __XNLINE__ Tensor &operator-=(Tensor &t){ f -= t.f; return *this; }
    __BOTH__ __XNLINE__ F32    operator+(Tensor &t)  { return f + t.f; }
    __BOTH__ __XNLINE__ F32    operator-(Tensor &t)  { return f - t.f; }
    __BOTH__ __XNLINE__ F32    operator*(Tensor &t)  { return f * t.f; }
    __BOTH__ __XNLINE__ F32    operator/(Tensor &t)  { return f / t.f; }
    __BOTH__ __XNLINE__ F32    operator%(Tensor &t)  { return fmod(f, t.f); }
	///
    /// tensor logical ops
    ///
    __BOTH__ __XNLINE__ bool   operator<(Tensor &t)  { return (f - t.f) <  -DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator>(Tensor &t)  { return (f - t.f) >   DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator<=(Tensor &t) { return (f - t.f) <= -DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator>=(Tensor &t) { return (f - t.f) >=  DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator==(Tensor &t) { return fabs(f - t.f) <  DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator!=(Tensor &t) { return fabs(f - t.f) >= DU_EPS; }
	///
    /// float arithmetics
    ///
    __BOTH__ __XNLINE__ Tensor &operator+=(F32 f0)   { f += f0; r = 0; return *this; }
    __BOTH__ __XNLINE__ Tensor &operator-=(F32 f0)   { f -= f0; r = 0; return *this; }
    __BOTH__ __XNLINE__ Tensor &operator*=(F32 f0)   { f *= f0; r = 0; return *this; }
    __BOTH__ __XNLINE__ Tensor &operator/=(F32 f0)   { f /= f0; r = 0; return *this; }
    ///
    /// float logical ops
    ///
    __BOTH__ __XNLINE__ bool   operator<(F32 f0)     { return (f - f0) <  -DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator>(F32 f0)     { return (f - f0) >   DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator>=(F32 f0)    { return (f - f0) >=  DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator==(F32 f0)    { return fabs(f - f0)  <  DU_EPS; }
    __BOTH__ __XNLINE__ bool   operator!=(F32 f0)    { return fabs(f - f0)  >= DU_EPS; }
} DU;
#endif // TEN4_SRC_TENSOR_H_
