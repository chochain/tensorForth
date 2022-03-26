/*! @file
  @brief
  cueForth Utility functions

  <pre>
  Copyright (C) 2021- GreenII

  This file is distributed under BSD 3-Clause License.
  </pre>
*/
#ifndef CUEF_SRC_UTIL_H_
#define CUEF_SRC_UTIL_H_
#include <stdint.h>
#include <stddef.h>

#define ALIGN4(sz)          ((sz) + (-(sz) & 0x3))
#define ALIGN8(sz)          ((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)         ((sz) + (-(sz) & 0xf))

#ifdef __cplusplus
extern "C" {
#endif

uint32_t hbin_to_u32(const void *bin);
uint16_t hbin_to_u16(const void *bin);

#if defined(__CUDACC__)
#define __GPU__      __device__
    
__GPU__ uint32_t     bin_to_u32(const void *bin);
__GPU__ uint16_t     bin_to_u16(const void *bin);

__GPU__ void         u16_to_bin(uint16_t s, const void *bin);
__GPU__ void         u32_to_bin(uint32_t l, const void *bin);

__GPU__ void         *d_memcpy(void *d, const void *s, size_t n);
__GPU__ void         *d_memset(void *d, int c, size_t n);
__GPU__ int          d_memcmp(const void *s1, const void *s2, size_t n);

__GPU__ int          d_strlen(const char *s, bool raw);
__GPU__ void         d_strcpy(char *d, const char *s);
__GPU__ int          d_strcmp(const char *s1, const char *s2);
__GPU__ int          d_strcasecmp(const char *s1, const char *s2);
__GPU__ char*        d_strchr(const char *s,  const char c);
__GPU__ char*        d_strcat(char *d,  const char *s);
__GPU__ char*        d_strcut(const char *s, int n);         // take n utf8 chars from the string
    
__GPU__ int          d_itoa(int v, char *s, int base=10);
__GPU__ long         d_strtol(const char *s, char **p, int base=10);
__GPU__ double       d_strtof(const char *s, char **p);
__GPU__ int          d_hash(const char *s);
    
// memory util
#define MEMCPY(d,s,n)   memcpy((char*)(d), (const char*)(s), (size_t)(n))
#define MEMSET(d,v,n)   memset((char*)(d), (int)(v), (size_t)(n))
#define MEMCMP(d,s,n)   d_memcmp((const char*)(d), (const char*)(s), (size_t)(n))
// string util
#define STRLEN(s)       d_strlen((const char*)(s), false)
#define STRLENB(s)      d_strlen((const char*)(s), true)
#define STRCPY(d,s)     MEMCPY((char*)(d), (const char*)(s), STRLENB(s)+1)
#define STRCMP(d,s)     MEMCMP((const char*)(d), (const char*)(s), STRLENB(s))
#define STRCASECMP(d,s) d_strcasecmp((const char*)(d), (const char*)(s))
#define STRCHR(d,c)     d_strchr((char*)(d), (const char)(c))
#define STRCAT(d,s)     d_strcat((char*)(d), (const char*)(s))
#define STRCUT(d,n)     d_strcut((const char*)(d), (int)(n))
// conversion
#define ITOA(i,s,b)     d_itoa((int)(i), (char*)(s), (int)(b))
#define STRTOL(s,p,b)   d_strtol((const char*)(s), (char**)(p), (int)(b))
#define STRTOF(s,p)     d_strtof((const char*)(s), (char**)(p))
#define HASH(s)         d_hash((const char*)(s))

#else  // !defined(__CUDACC__)
#include <stdio.h>
    
#define MEMCPY(d,s,n)   memcpy(d,s,n)
#define MEMCMP(d,s,n)   memcmp(d,s,n)
#define MEMSET(d,v,n)   memset(d,v,n)

#define STRLEN(s)       (int)strlen((char*)s)
#define STRLENB(s)      STRLEN(s)
#define STRCPY(d,s)     strcpy(d,s)
#define STRCMP(s1,s2)   strcmp(s1,s2)
#define STRCASECMP(s1,s2) strcasecmp(s1,s2)
#define STRCHR(d,c)     strchr(d,c)
#define STRCAT(d,s)     strcat(d,s)
#define STRCUT(s,n)     substr(s,n)

#define HASH(s)         calc_hash(s)            // add implementation
#define ITOA(i,s,b)     (sprintf(s,"%d",i))
#define STRTOL(s,p,b)   strtol(s,p,b)
#define STRTOF(s,p)     strtof(s,p)

#endif // defined(__CUDACC__)

#ifdef __cplusplus
}
#endif
#endif // CUEF_SRC_UTIL_H_
