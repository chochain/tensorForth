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

#define ALIGN4(sz)			((sz) + (-(sz) & 0x3))
#define ALIGN8(sz) 			((sz) + (-(sz) & 0x7))
#define ALIGN16(sz)  		((sz) + (-(sz) & 0xf))

#ifdef __cplusplus
extern "C" {
#endif

__host__   unsigned long hbin_to_u32(const void *bin);
__host__   unsigned int  hbin_to_u16(const void *bin);

#if defined(__CUDACC__)

__device__ unsigned long bin_to_u32(const void *bin);
__device__ unsigned int	 bin_to_u16(const void *bin);
__device__ void			u16_to_bin(unsigned int s, const void *bin);
__device__ void			u32_to_bin(unsigned long l, const void *bin);

__device__ void    		*d_memcpy(void *d, const void *s, size_t n);
__device__ void    		*d_memset(void *d, int c, size_t n);
__device__ int     		d_memcmp(const void *s1, const void *s2, size_t n);

__device__ int  		d_strlen(const char *s, int raw);
__device__ int     		d_strcmp(const char *s1, const char *s2);
__device__ char*		d_strchr(const char *s,  const char c);
__device__ char*		d_strcat(char *d,  const char *s);
__device__ char*     	d_strcut(const char *s, int n);			// take n utf8 chars from the string
__device__ long			d_strtol(const char *s, char **p, size_t base);
__device__ double		d_strtof(const char *s, char **p);
__device__ int 			d_hash(const char *s);
// memory util
#define MEMCPY(d,s,n)   memcpy(d,s,n)
#define MEMSET(d,v,n)   memset(d,v,n)
#define MEMCMP(d,s,n)   d_memcmp(d,s,n)
// string util
#define STRLEN(s)		d_strlen((char*)(s), 0)
#define STRLENB(s)		d_strlen((char*)(s), 1)
#define STRCPY(d,s)		MEMCPY(d,s,STRLENB(s)+1)
#define STRCMP(d,s)    	MEMCMP(d,s,STRLENB(s))
#define STRCHR(d,c)     d_strchr((char*)d,c)
#define STRCAT(d,s)     d_strcat((char*)d, (char*)s)
#define STRCUT(d,n)		d_strcut((char*)d, (int)n)
// conversion
#define STRTOL(s, p, base)  d_strtol((char*)(s), (char**)(p), base)
#define STRTOF(s, p)		d_strtof((char*)(s), (char**)(p))
#define HASH(s)			    d_hash((char*)(s))

#else

#define MEMCPY(d,s,sz)  memcpy(d, s, sz)
#define MEMCMP(d,s,sz)  memcmp(d, s, sz)
#define MEMSET(d,v,sz)  memset(d, v, sz)

#define STRLEN(s)		strlen(s)
#define STRCPY(d,s)	  	strcpy(d, s)
#define STRCMP(s1,s2)   strcmp(s1, s2)
#define STRCHR(d,c)     strchr(d, c)
#define STRCAT(d,s)     strcat(d, s)
#define STRCUT(s,sz)	substr(s, sz)

#define HASH(s)			    calc_hash(s)			// add implementation
#define STRTOL(s, p, base)	strtol(s, p, base)
#define STRTOF(s, p)		strtof(s, p)

#endif	// defined(__CUDACC__)

#ifdef __cplusplus
}
#endif
#endif // CUEF_SRC_UTIL_H_
