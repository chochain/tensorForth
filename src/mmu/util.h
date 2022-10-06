/**
 * @file
 * @brief common utility functions interface
 *
 * <pre>Copyright (C) 2021- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_UTIL_H_
#define TEN4_SRC_UTIL_H_
#include <stdint.h>
#include <stddef.h>
///
///@name Alignment macros
///@{
#define ALIGN2(sz)          ((sz) + (sz & 0x1))
#define ALIGN4(sz)          ((sz) + (-(sz) & 0x3))
#define ALIGN8(sz)          ((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)         ((sz) + (-(sz) & 0xf))
///@}
#ifdef __cplusplus
extern "C" {
#endif

uint32_t hbin_to_u32(const void *bin);
uint16_t hbin_to_u16(const void *bin);
#if defined(__CUDACC__)
///
///@name Endianess conversion
///@{
#define __KERN__     __global__
#define __GPU__      __device__
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
#define d_memset(t,s,n) memset(t,s,n)
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
///@}
///@name Tensor ops (kernel mode)
///@{
__KERN__ void        k_copy(float *src, float *dst, int sz);  ///< Note: (src, dst)
__KERN__ void        k_transpose(float *src, float *dst, int n, int h, int w); ///< Note: (src, dst), TODO: CDP
__KERN__ void        k_identity(float *t, int n, int h, int w, int c);
///@}
///==========================================================================
///@name Unified memory ops
///@{
#define MEMCPY(t,s,n)   d_memcpy((void*)(t), (void*)(s), (size_t)(n))       /** TODO: cudaMemcpyAsyn */
#define MEMSET(t,v,n)   d_memset((void*)(t), (int)(v), (size_t)(n))
#define MEMCMP(t,s,n)   d_memcmp((const char*)(t), (const char*)(s), (size_t)(n))
///@}
///@name Unified string ops
///@{
#define STRLEN(s)       d_strlen((const char*)(s), false)
#define STRLENB(s)      d_strlen((const char*)(s), true)
#define STRCPY(t,s)     d_strcpy((char*)(t), (const char*)(s))
#define STRCMP(t,s)     d_strcmp((const char*)(t), (const char*)(s))
#define STRCASECMP(t,s) d_strcasecmp((const char*)(t), (const char*)(s))
#define STRCHR(t,c)     d_strchr((char*)(t), (const char)(c))
#define STRCAT(t,s)     d_strcat((char*)(t), (const char*)(s))
#define STRCUT(s,n)     d_strcut((const char*)(s), (int)(n))
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
#define MEMSET(t,v,n)   memset(t,v,n)
#define MEMCMP(t,s,n)   memcmp(t,s,n)
///@}
///@name Unified string ops
///@{
#define STRLEN(s)       (int)strlen((char*)s)
#define STRLENB(s)      STRLEN(s)
#define STRCPY(t,s)     strcpy(t,s)
#define STRCMP(t,s)     strcmp(t,s)
#define STRCASECMP(t,s) strcasecmp(t,s)
#define STRCHR(t,c)     strchr(t,c)
#define STRCAT(t,s)     strcat(t,s)
#define STRCUT(s,n)     substr(s,n)
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
#endif // TEN4_SRC_UTIL_H_
