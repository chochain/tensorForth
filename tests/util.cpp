/*! @file
  @brief
  cueForth Utilities functions
    Memory cpy/set/cmp
    String hasher

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#define __CUDACC__    1
#define __device__

#include "../src/util.h"
#include <math.h>

typedef int           WORD;
#define WSIZE   	  (sizeof(WORD))
#define	WMASK		  (WSIZE-1)

#define DYNA_HASH_THRESHOLD     128
#define HASH_K 					1000003

uint32_t
hbin_to_u32(const void *bin)
{
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

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
uint16_t
hbin_to_u16(const void *bin)
{
    uint16_t x = *((uint16_t *)bin);
    return ((x & 0xff) << 8) | ((x >> 8) & 0xff);
}

__GPU__ void
_next_utf8(char **sp)
{
	char c = **sp;
	int  b = 0;
	if      (c>0 && c<=127) 		b=1;
	else if ((c & 0xE0) == 0xC0) 	b=2;
	else if ((c & 0xF0) == 0xE0) 	b=3;
	else if ((c & 0xF8) == 0xF0) 	b=4;
	else *sp=NULL;					// invalid utf8

	*sp+=b;
}

__GPU__ int
_loop_hash(const char *str, int bsz)
{
	// a simple polynomial hashing algorithm
	int h = 0;
    for (int i=0; i<bsz; i++) {
        h = h * HASH_K + str[i];
    }
    return h;
}

#if CUEF_ENABLE_CDP
//================================================================
/*! Calculate hash value

  @param  str	Target string.
  @return int	Symbol value.
*/
__KERN__ void
_dyna_hash(int *hash, const char *str, int sz)
{
	int x = threadIdx.x;									// row-major
	int m = __ballot_sync(0xffffffff, x<sz);				// ballot_mask
	int h = x<sz ? str[x] : 0;								// move to register

	for (int n=16; x<sz && n>0; n>>=1) {
		h += HASH_K*__shfl_down_sync(m, h, n);				// shuffle down
	}
	if (x==0) *hash += h;
}

__KERN__ void
_dyna_hash2d(int *hash, const char *str, int bsz)
{
	auto blk = cg::this_thread_block();						// C++11

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
#endif // CUEF_ENABLE_CDP
__GPU__ int _warp_h[32];			// each thread takes a slot
__GPU__ int
_hash(const char *str, int bsz)
{
	return 0;
}

//================================================================
/*!@brief
  little endian to big endian converter

  @param  s	Pointer of memory.
  @return	32bit unsigned value.
*/
__GPU__ uint32_t
bin_to_u32(const void *s)
{
#if CUEF_32BIT_ALIGN_REQUIRED
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

  @param  s	Pointer of memory.
  @return	16bit unsigned value.
*/
__GPU__ uint16_t
bin_to_u16(const void *s)
{
#if CUEF_32BIT_ALIGN_REQUIRED
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
u16_to_bin(uint16_t s, char *bin)
{
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
u32_to_bin(uint32_t l, char *bin)
{
    *bin++ = (l >> 24) & 0xff;
    *bin++ = (l >> 16) & 0xff;
    *bin++ = (l >> 8) & 0xff;
    *bin   = l & 0xff;
}

/* memcpy generic C-implementation,
 *
 *   TODO: alignment of ss is still
 */
__GPU__ void*
d_memcpy(void *d, const void *s, size_t n)
{
	if (n==0 || d==s) return d;

	char *ds = (char*)d, *ss = (char*)s;
	size_t t = (uintptr_t)ss;								// take low bits

	if ((uintptr_t)ds < (uintptr_t)ss) {					// copy forward
		if ((t | (uintptr_t)ds) & WMASK) {
			int i = (((t ^ (uintptr_t)ds) & WMASK) || (n < WSIZE))		// align operands
				? n
				: WSIZE - (t & WMASK);
			n -= i;
			for (; i; i--) *ds++ = *ss++;					// leading bytes
		}
		for (int i=n/WSIZE; i; i--) { *(WORD*)ds=*(WORD*)ss; ds+=WSIZE; ss+=WSIZE; }
		for (int i=n&WMASK; i; i--) *ds++ = *ss++;			// trailing bytes
	}
	else {													// copy backward
		ss += n;
		ds += n;
		if ((t | (uintptr_t)ds) & WMASK) {
			int i = (((t ^ (uintptr_t)ds) & WMASK) || (n <= WSIZE))
				? n
				: t & WMASK;
			n -= i;
			for (; i; i--) *--ds = *--ss;					// leading bytes
		}
		for (int i=n/WSIZE; i; i--) { ss-=WSIZE; ds-=WSIZE; *(WORD*)ds=*(WORD*)ss; }
		for (int i=n&WMASK; i; i--) *--ds = *--ss;
	}
	return d;
}

__GPU__ void*
d_memset(void *d, int c, size_t n)
{
    char *s = (char*)d;

    /* Fill head and tail with minimal branching. Each
     * conditional ensures that all the subsequently used
     * offsets are well-defined and in the dest region. */

    if (!n) return d;
    s[0] = s[n-1] = c;
    if (n <= 2) return d;
    s[1] = s[n-2] = c;
    s[2] = s[n-3] = c;
    if (n <= 6) return d;
    s[3] = s[n-4] = c;
    if (n <= 8) return d;

    /* Advance pointer to align it at a 4-byte boundary,
     * and truncate n to a multiple of 4. The previous code
     * already took care of any head/tail that get cut off
     * by the alignment. */

    size_t k = -(uintptr_t)s & 3;
    s += k;
    n -= k;
    n &= -4;			// change of sign???
    n /= 4;

    uint32_t *ws = (uint32_t *)s;
    uint32_t  wc = c & 0xFF;
    wc |= ((wc << 8) | (wc << 16) | (wc << 24));

    /* Pure C fallback with no aliasing violations. */
    for (; n; n--) *ws++ = wc;

    return d;
}

__GPU__ int
d_memcmp(const void *s1, const void *s2, size_t n)
{
	char *p1=(char*)s1, *p2=(char*)s2;
	for (; n; n--, p1++, p2++) {
		if (*p1 != *p2) return *p1 - *p2;
	}
	return 0;
}

__GPU__ int
d_strlen(const char *str, int raw)
{
	int  n  = 0;
	char *s = (char*)str;
	for (int i=0; s && *s!='\0'; i++, n++) {
		_next_utf8(&s);
	}
	return (s && raw) ? s - str : n;
}

__GPU__ void
d_strcpy(char *d, const char *s)
{
    d_memcpy(d, s, STRLENB(s)+1);
}

__GPU__ int
d_strcmp(const char *s1, const char *s2)
{
    return d_memcmp(s1, s2, STRLENB(s1));
}

__GPU__ char*
d_strchr(const char *s, const char c)
{
	char *p = (char*)s;
    for (; p && *p!='\0'; p++) {
    	if (*p==c) return p;
    }
    return NULL;
}

__GPU__ char*
d_strcat(char *d, const char *s)
{
	d_memcpy(d+STRLENB(d), s, STRLENB(s)+1);
    return d;
}

__GPU__ char*
d_strcut(const char *s, int n)
{
	char *p = (char*)s;
	for (int i=0; n && i<n && p && *p!='\0'; i++) {
		_next_utf8(&p);
	}
	return p;
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

  @param  s	source string.
  @param  base	n base.
  @return	result.
*/
__GPU__ long
d_strtol(const char *s, char** p, int base)
{
    long ret  = 0;
    bool sign = 0;

REDO:
    switch(*s) {
    case '-': sign = 1;		// fall through.
    case '+': s++;	        break;
    case ' ': s++;          goto REDO;
    }
    *p = NULL;
    char ch;
    int  n;
    while ((ch = *s++) != '\0') {
        *p = (char*)s;
        if      ('a' <= ch) 			 n = ch - 'a' + 10;
        else if ('A' <= ch) 			 n = ch - 'A' + 10;
        else if ('0' <= ch && ch <= '9') n = ch - '0';
        else break;
        if (n >= base) break;

        ret = ret * base + n;
    }
    return (sign) ? -ret : ret;
}

__GPU__ double
d_strtof(const char *s, char** p)
{
    int sign = 1, esign = 1, state=0;
    int r = 0, e = 0;
    long v = 0L, f = 0L;

    while ((*s<'0' || *s>'9') && *s!='+' && *s!='-') s++;

    if (*s=='+' || *s=='-') sign = *s++=='-' ? -1 : 1;
    
    *p = NULL;
    while (*s!='\0' && *s!='\n' && *s!=' ' && *s!='\t') {
    	if (state==0 && *s>='0' && *s<='9') {	    // integer
    		v = (*s - '0') + v * 10;
    	}
    	else if (state==1 && *s>='0' && *s<='9') {	// decimal
            f = (*s - '0') + f * 10;
            r--;
        }
    	else if (state==2) {						// exponential
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
d_hash(const char *s)
{
	return _hash(s, STRLENB(s));
}
