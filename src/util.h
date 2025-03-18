/**
 * @file
 * @brief common utility functions interface
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __UTIL_H_
#define __UTIL_H_
#include <stdint.h>
#include <stddef.h>
#include "ten4_config.h"
///@}
///@name Alignment macros
///@{
#define ALIGN2(sz)  ((sz) + (sz & 0x1))
#define ALIGN4(sz)  ((sz) + (-(sz) & 0x3))
#define ALIGN8(sz)  ((sz) + (-(sz) & 0x7))
#define ALIGN16(sz) ((sz) + (-(sz) & 0xf))
///@}
///@name cross platform floating-point ALU support (see nvcc -use_fast_math flag)
///@{
typedef enum {
    /// 1-operand arithmetic ops
    ABS = 0,
    NEG,
    EXP,
    LN,
    LOG,
    TANH,
    RELU,
    SIGM,
    SQRT,
    RCP,
    SAT,
    IDEN,
    /// 1-operand + a constant
    FILL,
    GFILL,
    SCALE,
    POW,
    /// 2-operand ops
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    MAX,
    MIN,
    MUL2,
    MOD2
} math_op;

#define MATH_OP "abs","neg","exp","ln","log","tanh","relu","sigmoid","sqrt","rcp","sat","iden","fill","gfill","scale","pow","+","-","*","/","mod","max","min","mul2","mod2"

#define ABS(d)      (fabsf(d))                 /**< absolute value         */
#define NEG(d)      (-d)                       /**< negate                 */
#define EXP(d)      (__expf(d))                /**< exponential(float)     */
#define LN(d)       (__logf(d))                /**< natural logrithm       */
#define LOG(d)      (__log10f(d))              /**< log10                  */
#define TANH(d)     (atanhf(d))                /**< tanh(float)            */
#define RELU(d)     (MAX(0.0, d))              /**< relu(float)            */
#define SIGMOID(d)  (RCP(1.0+EXP(-(d))))       /**< sigmoid(float)         */
#define SQRT(d)     (__fsqrt_rn(d))            /**< square root            */
#define RCP(x)      (__frcp_rn(x))             /**< reciprocol 1/x         */
#define SAT(d)      (__saturatef(d))           /**< clamp into [0.0..1.0]  */
#define POW(d,e)    (__powf(d,e))              /**< power d^(e)            */
#define ADD(x,y)    (__fadd_rn(x,y))           /**< addition               */
#define SUB(x,y)    (__fsub_rn(x,y))           /**< addition               */
#define MUL(x,y)    (__fmul_rn(x,y))           /**< multiplication         */
#define DIV(x,y)    (__fdiv_rn(x,y))           /**< division               */
#define MOD(t,n)    (fmodf(t,n))               /**< fmod two floats        */
#define MAX(x,y)    (fmax(x,y))                 /**< maximum of the two     */
#define MIN(x,y)    (fmin(x,y))                 /**< minimum of the two     */
#define MUL2(x2,y2) (__dmul_rn(x2,y2))         /**< double precision mul   */
#define MOD2(x2,y2) (fmod(x2,y2))              /**< double precision mod   */
///@}
#define __HOST__     __host__
#define __KERN__     __global__
#define __GPU__      __device__

#ifdef __cplusplus
extern "C" {
#endif

uint32_t hbin_to_u32(const void *bin);
uint16_t hbin_to_u16(const void *bin);
#if defined(__CUDACC__)
///
///@name Endianess conversion
///@{
__GPU__ uint32_t     bin_to_u32(const void *bin);
__GPU__ uint16_t     bin_to_u16(const void *bin);

__GPU__ void         u16_to_bin(uint16_t s, const void *bin);
__GPU__ void         u32_to_bin(uint32_t l, const void *bin);
///@}
///@name Memory ops
///@{
//__GPU__ void         d_memcpy(void *t, const void *s, size_t n);
//__GPU__ void         d_memset(void *t, int c, size_t n);
#define d_memcpy(t,s,n) memcpy(t,s,n)
#define d_memset(t,c,n) memset(t,c,n)
__GPU__ int          d_memcmp(const void *t, const void *s, size_t n);
///@}
///@name String ops
///@{
__GPU__ int          d_strlen(const char *s, bool raw);
__GPU__ void         d_strcpy(char *t, const char *s);
__GPU__ int          d_strcmp(const char *t, const char *s);
__GPU__ int          d_strcasecmp(const char *t, const char *s);
__GPU__ char*        d_strchr(const char *s,  const char c);
__GPU__ char*        d_strcat(char *t,  const char *s);
__GPU__ char*        d_strcut(const char *s, int n);                     // take n utf8 chars from the string
///@}
///@name Numeric conversion
///@{
__GPU__ int          d_itoa(int v, char *s, int base=10);
__GPU__ long         d_strtol(const char *s, char **p, int base=10);
__GPU__ double       d_strtof(const char *s, char **p);
__GPU__ int          d_hash(const char *s);
__GPU__ float        d_sum(float *src, long numel);
__GPU__ float        d_var_sq(float *src, float avg, long numel);
///@}
///@name Tensor ops (kernel mode)
///@{
__KERN__ void        k_sum4(float *src, float *sum, long numel);
__KERN__ void        k_var4(float *src, float *avg, float *var, long numel);
__KERN__ void        k_copy(float *src, float *dst, long n);                   ///< Note: (src, dst)
__KERN__ void        k_transpose(float *src, float *dst, int h, int w);        ///< Note: (src, dst), TODO: CDP
__KERN__ void        k_identity(float *t, int h, int w);
__KERN__ void        k_math(math_op op, float *dst, float v, long n);          ///< tensor math ops
__KERN__ void        k_ts_op(math_op op, float *A, float v, float *O, long n); ///< tensor-scalar ops
__KERN__ void        k_tt_op(math_op op, float *A, float *B, float *O, long n);///< tensor-tensor ops
__KERN__ void        k_bce(float *O, float *T, long n);
///@}
///==========================================================================
///@name Unified memory ops
///@{
#define MEMCPY(t,s,n)   d_memcpy((void*)(t), (void*)(s), (size_t)(n))       /** TODO: cudaMemcpyAsyn */
#define MEMSET(t,c,n)   d_memset((void*)(t), (int)(c), (size_t)(n))
#define MEMCMP(t,s,n)   d_memcmp((const char*)(t), (const char*)(s), (size_t)(n))
///@}
///@name Unified string ops
///@{
#define STRLEN(s)       d_strlen((const char*)(s), false)
#define STRLENB(s)      d_strlen((const char*)(s), true)
#define STRCPY(t,s)     d_strcpy((char*)(t), (const char*)(s))
#define STRCHR(t,c)     d_strchr((char*)(t), (c))
#define STRCAT(t,s)     d_strcat((char*)(t), (const char*)(s))
#define STRCUT(s,n)     d_strcut((const char*)(s), (int)(n))
#if T4_CASE_SENSITIVE    
#define STRCMP(t,s)     d_strcmp((const char*)(t), (const char*)(s))
#else  // T4_CASE_SENSITIVE
#define STRCMP(t,s)     d_strcasecmp((const char*)(t), (const char*)(s))
#endif // T4_CASE_SENSITIVE    
///@}
///@name Unified numeric conversion ops
///@{
#define ITOA(i,s,b)     d_itoa((int)(i), (char*)(s), (int)(b))
#define STRTOL(s,p,b)   d_strtol((const char*)(s), (char**)(p), (int)(b))
#define STRTOF(s,p)     d_strtof((const char*)(s), (char**)(p))
#define HASH(s)         d_hash((const char*)(s))
///@}
#else  // !defined(__CUDACC__)
#include <stdio.h>
///
///@name Unified memory ops
///@{
#define MEMCPY(t,s,n)   memcpy(t,s,n)
#define MEMSET(t,c,n)   memset(t,c,n)
#define MEMCMP(t,s,n)   memcmp(t,s,n)
///@}
///@name Unified string ops
///@{
#define STRLEN(s)       (int)strlen((char*)s)
#define STRLENB(s)      STRLEN(s)
#define STRCPY(t,s)     strcpy(t,s)
#define STRCHR(t,c)     strchr(t,c)
#define STRCAT(t,s)     strcat(t,s)
#define STRCUT(s,n)     substr(s,n)
#if T4_CASE_SENSITIVE
#define STRCMP(t,s)     strcmp(t,s)
#else  // !T4_CASE_SENSITIVE
#define STRCMP(t,s)     strcasecmp(t,s)
#endif // T4_CASE_SENSITIVE
///@}
///@name Unified numeric conversion ops
///@{
#define ITOA(i,s,b)     (sprintf(s,"%d",i))
#define STRTOL(s,p,b)   strtol(s,p,b)
#define STRTOF(s,p)     strtof(s,p)
#define HASH(s)         calc_hash(s)
///@}
#endif // defined(__CUDACC__)

#ifdef __cplusplus
}
#endif
#endif // __UTIL_H_
