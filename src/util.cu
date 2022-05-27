/**
 * @file
 * @brief tensorForth Utilities functions
 *  + Memory cpy/set/cmp
 *  + String hasher
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "ten4_types.h"
#include "util.h"

#if TEN4_ENABLE_CDP
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif // TEN4_ENABLE_CDP

typedef int           WORD;
#define WSIZE         (sizeof(WORD))
#define WMASK         (WSIZE-1)

#define DYNA_HASH_THRESHOLD     128
#define HASH_K                  1000003

uint32_t
hbin_to_u32(const void *bin) {
    uint32_t x = *((uint32_t*)bin);
    return
        ((x & 0xff)   << 24) |
        ((x & 0xff00) << 8)  |
        ((x >> 8)  & 0xff00) |
        ((x >> 24) & 0x00ff);
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param    bin Pointer of memory.
  @return   16-bit unsigned value.
*/
uint16_t
hbin_to_u16(const void *bin) {
    uint16_t x = *((uint16_t *)bin);
    return ((x & 0xff) << 8) | ((x >> 8) & 0xff);
}

__GPU__ void
_next_utf8(char **sp) {
    char c = **sp;
    int  b = 0;
    if      (c>0 && c<=127)         b=1;
    else if ((c & 0xE0) == 0xC0)    b=2;
    else if ((c & 0xF0) == 0xE0)    b=3;
    else if ((c & 0xF8) == 0xF0)    b=4;
    else *sp=NULL;                  // invalid utf8

    *sp+=b;
}

__GPU__ int
_loop_hash(const char *str, int bsz) {
    // a simple polynomial hashing algorithm
    int h = 0;
    for (int i=0; i<bsz; i++) {
        h = h * HASH_K + str[i];
    }
    return h;
}

#if TEN4_ENABLE_CDP
//================================================================
/*! Calculate hash value

  @param  str   Target string.
  @return int   Symbol value.
*/
__KERN__ void
_dyna_hash(int *hash, const char *str, int sz) {
    int x = threadIdx.x;                                    // row-major
    int m = __ballot_sync(0xffffffff, x<sz);                // ballot_mask
    int h = x<sz ? str[x] : 0;                              // move to register

    for (int n=16; x<sz && n>0; n>>=1) {
        h += HASH_K*__shfl_down_sync(m, h, n);              // shuffle down
    }
    if (x==0) *hash += h;
}

__KERN__ void
_dyna_hash2d(int *hash, const char *str, int bsz) {
    auto blk = cg::this_thread_block();                     // C++11

    extern __shared__ int h[];

    int x = threadIdx.x;
    int y = threadIdx.y*blockDim.x;
    h[x+y] = 0;

    for (int n=0; n<blockDim.y; n++) {
        if ((x+y)<bsz) h[y] += HASH_K*h[y+n*blockDim.x];
        blk.sync();
    }
    for (int n=blockDim.x>>1, off=n+y; n>0; off=(n>>=1)+y) {
        if (x<n && (x+off)<bsz) h[x+y] += HASH_K*h[x+off];
        blk.sync();
    }
    *hash = h[0];
}
#endif // TEN4_ENABLE_CDP
__GPU__ int _warp_h[32];            // each thread takes a slot
__GPU__ int
_hash(const char *str, int bsz) {
    if (bsz < DYNA_HASH_THRESHOLD) return _loop_hash(str, bsz);

    int x  = threadIdx.x;
    int *h = &_warp_h[x];   *h=0;                           // each calling thread takes a slot

#if CUDA_ENABLE_CDP
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);  // wrapper overhead ~= 84us

    for (int i=0; i<bsz; i+=32) {
        _dyna_hash<<<1,32,0,st>>>(h, &str[i], bsz-i);
        GPU_SYNC();                                         // sync all children threads
    }

    dim3 xyz(32, (bsz>>5)+1, 0);
    int  blk = bsz+(-bsz&0x1f);
    _dyna_hash2d<<<1,xyz,blk*sizeof(int)>>>(h, str, bsz);

    GPU_SYNC();

    cudaStreamDestroy(st);
#endif // CUDA_ENABLE_CDP

    return *h;
}

//================================================================
/*!@brief
  little endian to big endian converter

  @param  s Pointer of memory.
  @return   32bit unsigned value.
*/
__GPU__ uint32_t
bin_to_u32(const void *s) {
#if TEN4_32BIT_ALIGN_REQUIRED
    char *p = (char*)s;
    return (uint32_t)(p[0]<<24) | (p[1]<<16) |  (p[2]<<8) | p[3];
#else
    uint32_t x = *((uint32_t*)s);
    return (x << 24) | ((x & 0xff00) << 8) | ((x >> 8) & 0xff00) | (x >> 24);
#endif
}

//================================================================
/*!@brief
  Get 16bit value from memory big endian.

  @param  s Pointer of memory.
  @return   16bit unsigned value.
*/
__GPU__ uint16_t
bin_to_u16(const void *s) {
#if TEN4_32BIT_ALIGN_REQUIRED
    char *p = (char*)s;
    return (uint16_t)(p[0]<<8) | p[1];
#else
    uint16_t x = *((uint16_t*)s);
    return (x << 8) | (x >> 8);
#endif
}

/*!@brief
  Set 16bit big endian value from memory.

  @param  s Input value.
  @param  bin Pointer of memory.
  @return sizeof(U16).
*/
__GPU__ void
u16_to_bin(uint16_t s, char *bin) {
    *bin++ = (s >> 8) & 0xff;
    *bin   = s & 0xff;
}

/*!@brief
  Set 32bit big endian value from memory.

  @param  l Input value.
  @param  bin Pointer of memory.
  @return sizeof(U32).
*/
__GPU__ void
u32_to_bin(uint32_t l, char *bin) {
    *bin++ = (l >> 24) & 0xff;
    *bin++ = (l >> 16) & 0xff;
    *bin++ = (l >> 8) & 0xff;
    *bin   = l & 0xff;
}
/*
__GPU__ void
d_memcpy(void *t, const void *s, size_t n) {
    char *p1=(char*)t, *p0=(char*)s;
    for (; n; n--) *p1++ = *p0++;
}

__GPU__ void
d_memset(void *t, int c, size_t n) {
    char *p1=(char*)t;
    for (; n; n--) *p1++ = (char)c;
}
*/
__GPU__ int
d_memcmp(const void *t, const void *s, size_t n) {
    char *p1=(char*)t, *p0=(char*)s;
    for (; n; p1++, p0++, n--) {
        if (*p1 != *p0) return *p1 - *p0;
    }
    return 0;
}

__GPU__ int
d_strlen(const char *s, bool raw) {
    char *p0 = (char*)s;
    int  n;
    for (n=0; p0 && *p0; n++) {
        _next_utf8(&p0);
    }
    return (p0 && raw) ? p0 - s : n;
}

__GPU__ void
d_strcpy(char *t, const char *s) {
    char *p1=(char*)t, *p0=(char*)s;
    while (*p0) *p1++ = *p0++;
}

__GPU__ int
d_strcmp(const char *t, const char *s) {
    char *p1=(char*)t, *p0=(char*)s;
    for (; *p1 && *p0 && *p1==*p0; p1++, p0++);
    return *p1 - *p0;
}

__GPU__ int
d_strcasecmp(const char *t, const char *s) {
    char *p1=(char*)t, *p0=(char*)s;
    for (; *p1 && *p0; p1++, p0++) {
        char c = *p1 & 0x7f;
        if (c < 0x41 || c > 0x5a) {
            if (*p1 != *p0) break;
        }
        else if ((*p1 & 0x5f) != (*p0 & 0x5f)) break;
    }
    return *p1 - *p0;
}

__GPU__ char*
d_strchr(const char *s, const char c) {
    char *p = (char*)s;
    for (; p && *p!='\0'; p++) {
        if (*p==c) return p;
    }
    return NULL;
}

__GPU__ char*
d_strcat(char *t, const char *s) {
    d_memcpy(t+STRLENB(t), s, STRLENB(s)+1);
    return t;
}

__GPU__ char*
d_strcut(const char *s, int n) {
    char *p0 = (char*)s;
    for (int i=0; n && i<n && p0 && *p0!='\0'; i++) {
        _next_utf8(&p0);
    }
    return p0;
}

__GPU__ int
d_itoa(int v, char *s, int base) {
    char b[36], *p = &b[35];
    bool     sign = base==10 && v<0;
    uint32_t x    = sign ? -v : v;
    *p-- = '\0';
    do {
        uint32_t dx = x % base;
        *p-- = (char)(dx>9 ? (dx-10)+'A' : dx+'0');
        x /= base;
    } while (x != 0);
    if (sign) *p--='-';
    x = &b[35] - p;
    d_memcpy(s, (p+1), x);
    return x-1;
}

//================================================================
/*!@brief

  convert ASCII string to integer Guru version

  @param  s source string.
  @param  p return pointer
  @param  base  n base.
  @return   result.
*/
__GPU__ long
d_strtol(const char *s, char** p, int base) {
    long ret  = 0;
    bool sign = 0;

REDO:
    switch(*s) {
    case '-': sign = 1;     // fall through.
    case '+': s++;          break;
    case ' ': s++;          goto REDO;
    }
    *p = NULL;
    char ch;
    int  n;
    while ((ch = *s++) != '\0') {
        *p = (char*)s;
        if      ('a' <= ch)              n = ch - 'a' + 10;
        else if ('A' <= ch)              n = ch - 'A' + 10;
        else if ('0' <= ch && ch <= '9') n = ch - '0';
        else break;
        if (n >= base) break;

        ret = ret * base + n;
    }
    return (sign) ? -ret : ret;
}

__GPU__ double
d_strtof(const char *s, char** p) {
    int sign = 1, esign = 1, state=0;
    int r = 0, e = 0;
    long v = 0L, f = 0L;

    while ((*s<'0' || *s>'9') && *s!='+' && *s!='-') s++;

    if (*s=='+' || *s=='-') sign = *s++=='-' ? -1 : 1;

    *p = NULL;
    while (*s!='\0' && *s!='\n' && *s!=' ' && *s!='\t') {
        if (state==0 && *s>='0' && *s<='9') {       // integer
            v = (*s - '0') + v * 10;
        }
        else if (state==1 && *s>='0' && *s<='9') {  // decimal
            f = (*s - '0') + f * 10;
            r--;
        }
        else if (state==2) {                        // exponential
            if (*s=='-') {
                esign = -1;
                s++;
            }
            if (*s>='0' && *s<='9') e = (*s - '0') + e * 10;
        }
        state = (*s=='e' || *s=='E') ? 2 : ((*s=='.') ? 1 : state);
        s++;
        *p = (char*)s;
    }
    return sign *
        (v + (f==0 ? 0.0f : f * exp10((double)r))) *
        (e==0 ? 1.0f : exp10((double)esign * e));
}

__GPU__ int
d_hash(const char *s) {
    return _hash(s, STRLENB(s));
}
