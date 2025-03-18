/** -*- c++ -*-
 * @file
 * @brief common utility functions implementation
 *  + Memory cpy/set/cmp
 *  + String hasher
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "util.h"

#if T4_DO_CDP
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif // T4_DO_CDP

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

#if T4_DO_CDP
//================================================================
/*! Calculate hash value

  @param  str   Target string.
  @return int   Symbol value.
*/
__GPU__ void
_dyna_hash(int *hash, const char *str, int sz) {
    int x = threadIdx.x;                                    // row-major
    int m = __ballot_sync(0xffffffff, x<sz);                // ballot_mask
    int h = x<sz ? str[x] : 0;                              // move to register

    for (int n=16; x<sz && n>0; n>>=1) {
        h += HASH_K*__shfl_down_sync(m, h, n);              // shuffle down
    }
    if (x==0) *hash += h;
}

__GPU__ void
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
#endif // T4_DO_CDP
__GPU__ int _warp_h[32];            // each thread takes a slot
__GPU__ int
_hash(const char *str, int bsz) {
    if (bsz < DYNA_HASH_THRESHOLD) return _loop_hash(str, bsz);

    int x  = threadIdx.x;
    int *h = &_warp_h[x];   *h=0;                           // each calling thread takes a slot

#if 0 && T4_DO_CDP
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);  // wrapper overhead ~= 84us

    for (int i=0; i<bsz; i+=32) {
        _dyna_hash<<<1,32,0,st>>>(h, &str[i], bsz-i);
        CDP_SYNC();                                         // sync all children threads
    }

    dim3 xyz(32, (bsz>>5)+1);
    int  blk = bsz+(-bsz&0x1f);
    _dyna_hash2d<<<1,xyz,blk*sizeof(int)>>>(h, str, bsz);

    CDP_SYNC();
    cudaStreamDestroy(st);
#endif // T4_DO_CDP

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
#if T4_ALIGN4
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
#if T4_ALIGN4
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
        char c = *p1 & 0x5f;
        if (c < 0x41 || c > 0x5a) {
            if (*p1 != *p0) break;
        }
        else if (c != (*p0 & 0x5f)) break;  // [a-zA-Z]
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

    while (*s==' ' || *s=='\t') s++;
    if (*s=='+' || *s=='-') sign = *s++=='-' ? -1 : 1;

    *p = (char*)s;      // init to not NULL
    char c;
    int  n;
    while ((c = *s++) != '\0') {
        *p = (char*)s;
        if      (c >='a')            n = c - 'a' + 10;  // abcdef
        else if (c >='A')            n = c - 'A' + 10;  // ABCDEF
        else if (c <='9' && c >='0') n = c - '0';       // 0~9
        else break;
        if (n >= base) break;

        ret = ret * base + n;
    }
    return (sign) ? -ret : ret;
}

__GPU__ double
d_strtof(const char *s, char** p) {
    int  sign = 1, esign = 1, state=0;
    int  r = 0,  e = 0;
    long v = 0L, f = 0L;
    auto digi = [](char c) { return c>='0' && c<='9'; };
    auto expo = [](char c) { return c=='e' || c=='E'; };
    auto done = [](char c) { return c=='\0' || c=='\n' || c==' ' || c=='\t'; };
    
//    printf("\nd_strtof(%s)\n", s);
    while (*s==' ' || *s=='\t') s++;
    if (*s=='+' || *s=='-') sign = *s++=='-' ? -1 : 1;

    *p = (char*)s;                                  // init to not NULL
    char c = *s;
    while (!done(c)) {
//        printf("\n\nc,st,v,f,e=%x,%d:%ld,%ld[%d],%d", c, state, v, f, r, e);
        if (state==0) {
            if (digi(c)) {                          // integer
                v = (c - '0') + v * 10;
            }
            else if (c=='.')  state = 1;
            else if (expo(c)) state = 2;
            else break;
        }
        else if (state==1) {
            if (digi(c)) {                          // decimal
                f = (c - '0') + f * 10;
                r--;                                // depth
            }
            else if (expo(c)) state = 2;
            else break;
        }
        else if (state==2) {                        // exponential
            if (c=='-') {
                esign = -1;
                c=*(++s);
            }
            if (digi(c)) e = (c - '0') + e * 10;
            else break;
        }
//        printf("\nc,st,v,f,e=%x,%d:%ld,%ld[%d],%d", c, state, v, f, r, e);
        c = *(++s);
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

__GPU__ float
d_sum(float *src, long numel) {                     ///< sum of T4_DIM_SQ threads
    __shared__ float _sum[T4_DIM_SQ>>5];            ///< warp sum
    ///
    /// sum up by stride
    ///
    auto const g   { cg::this_thread_block() };     /// total threads
    auto const tid { g.thread_rank() };             /// tid=thread_index().x 0~255
    auto stride_sum = [src, numel](long i0) {
        float v { 0.0f };
        for (long i=i0; i < numel; i+=T4_DIM_SQ) v += src[i];
        return v;
    };
    float sum { stride_sum(tid) };                  /// one sum per thread
    ///
    /// shuffle sum 32 to 1
    ///
    auto tp { cg::tiled_partition<32>(g) };
    auto shfl_sum = [](cg::thread_block_tile<32> tp, float v) {
        #pragma unroll
        for (int k = tp.size()>>1; k > 0; k >>= 1) {
            v += tp.shfl_down(v, k);
        }
        return v;
    };
    sum = shfl_sum(tp, sum);
    if (tp.thread_rank() == 0) _sum[tid >> 5] = sum; /// collection from each warp 
    g.sync();
    ///
    /// sum up all warps
    ///
    float v { 0.0f };
    #pragma unroll
    for (int i = 0; i < (T4_DIM_SQ>>5); i++) v += _sum[i];
    
    return v;
}

__GPU__ float
d_var_sq(float *src, float avg, long numel) {       ///< sum of T4_DIM_SQ threads
    __shared__ float _sum[T4_DIM_SQ>>5];            ///< warp sum
    ///
    /// sum up by stride
    ///
    auto stride_sum = [src, avg, numel](long i0) {  ///< stride sum
        float v { 0.0f };
        for (long i=i0; i < numel; i+=T4_DIM_SQ) {
            v += (src[i] - avg) * (src[i] - avg);
        }
        return v;
    };
    auto const g   { cg::this_thread_block() };     /// total threads
    auto const tid { g.thread_rank() };             /// tid=thread_index().x 0~255
    float sum { stride_sum(tid) };                  /// one sum per thread
    ///
    /// shuffle sum 32 to 1
    ///
    auto tp { cg::tiled_partition<32>(g) };
    auto shfl_sum = [](cg::thread_block_tile<32> tp, float v) {
        #pragma unroll
        for (int k = tp.size()>>1; k > 0; k >>= 1) {
            v += tp.shfl_down(v, k);
        }
        return v;
    };
    sum = shfl_sum(tp, sum);
    if (tp.thread_rank() == 0) _sum[tid >> 5] = sum; /// collection from each warp 
    g.sync();
    ///
    /// sum up all warps
    ///
    float v { 0.0f };
    #pragma unroll
    for (int i = 0; i < (T4_DIM_SQ>>5); i++) v += _sum[i];
    
    return v;
}
///> Tensor HW sum per N per channel
///
__KERN__ void
k_sum4(float *src, float *dst, long HW) {
    const long j  = (long)blockIdx.x*blockDim.x + threadIdx.x; ///< element index
    const int  c  = blockIdx.y, n = blockIdx.z, C = gridDim.y; ///< channel
    const long ns = HW * C * n;                                ///< batch slice index
    float vi = j < HW ? src[ns + j * C + c] : 0.0f;
    ///
    /// prefix sum every 32-threaded tile
    ///
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (tp.thread_rank() == 0) dst[C * n + c] = 0.0f;
    __syncthreads();
    
    auto shfl_sum = [](cg::thread_block_tile<32> tp, float v) {
        for (int k = 16; k > 0; k >>= 1) {
            v += tp.shfl_down(v, k);
        }
        return v;
    };
    vi = shfl_sum(tp, vi);
    ///
    /// sum up atomically (per channel, for batchnorm)
    /// slower than grid-stride loop when blocks are many
    ///
    if (tp.thread_rank() == 0) atomicAdd_block(&dst[C * n + c], vi);   ///< serialize sum
}
///> variance
///
__KERN__ void
k_var4(float *src, float*avg, float *var, long HW) {
    const long j  = (long)blockIdx.x * blockDim.x + threadIdx.x;  ///< element index
    const int  c  = blockIdx.y, n = blockIdx.z, C = gridDim.y;   ///< channel
    const long ns = HW * C * n;                                  ///< batch slice index
    float v0 = j < HW ? src[(long)C * j + ns + c] - avg[C * n + c] : 0.0f;
    float vi = v0 * v0;
    ///
    /// prefix sum every 32-threaded tile
    ///
    auto tp = cg::tiled_partition<32>(cg::this_thread_block());
    if (tp.thread_rank() == 0) var[C * n + c] = 0.0f;
    
    auto shfl_sum = [](cg::thread_block_tile<32> tp, float v) {
        for (int k = 16; k > 0; k >>= 1) {
            v += tp.shfl_down(v, k);
        }
        return v;
    };
    vi = shfl_sum(tp, vi);
    ///
    /// sum up atomically (per channel, for batchnorm)
    ///
    if (tp.thread_rank() == 0) atomicAdd_block(&var[C * n + c], vi);
}

__KERN__ void
k_copy(float *src, float *dst, long n) {                      ///< Note: (src, dst)
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        dst[j] = src[j];
    }
}
__KERN__ void
k_transpose(float *src, float *dst, int H, int W) {           ///< Note: (src, dst)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;      ///< W range 2G  * 1K = 2T,  U41
    const int i = blockIdx.y * blockDim.y + threadIdx.y;      ///< H range 65K * 1K = 65M, U26
    const int c = blockIdx.z, C = gridDim.z;                  ///< channel deep

    if (i < H && j < W && c < C) {
        dst[((long)H * j + i) * C + c] = src[((long)W * i + j) * C + c];
    }
}
__KERN__ void
k_identity(float *t, int H, int W) {                          ///< identity matrix (tensor)
    const float i01[2] = { 0.0f, 1.0f };
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z, C = gridDim.z;                  ///< channel deep

    if (i < H && j < W && c < C) {
        t[((long)W * i + j) * C + c] = i01[i==j];
    }
}

#define DU_LNX   1.0e-12                                      /* log clamp */
__KERN__ void
k_math(math_op op, float *A, float v, long n) {               ///< self modifying ops
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        float ak = A[j];                                      ///< cache value
        switch(op) {
        case ABS:   A[j] = ABS(ak);                   break;
        case NEG:   A[j] = NEG(ak);                   break;
        case EXP:   A[j] = EXP(ak);                   break;
        case LN:    A[j] = LN(MAX(ak, DU_LNX));       break;  /// * clamped
        case LOG:   A[j] = LOG(MAX(ak, DU_LNX));      break;  /// * clamped
        case TANH:  A[j] = TANH(ak);                  break;
        case RELU:  A[j] = RELU(ak);                  break;
        case SIGM:  A[j] = SIGMOID(ak);               break;
        case SQRT:  A[j] = SQRT(MAX(ak, 0.0));        break;  /// * guarded
        case RCP:   A[j] = RCP(ak);                   break;  /// 1/x
        case SAT:   A[j] = SAT(ak);                   break;  /// [0.0..1.0]
        case FILL:  A[j] = v;                         break;
        case GFILL: A[j] = v * j / n;                 break;  /// gradient fill
        case SCALE: A[j] *= v;                        break;
        case POW:   A[j] = POW(ak, v);                break;  /// x^v
        case ADD:   A[j] += v;                        break;
        case SUB:   A[j] -= v;                        break;
        case MUL:   A[j] *= v;                        break;
        case DIV:   A[j] /= v;                        break;
        default: printf("k_math op=%d not supported\n", op);
        }
    }
}
///
/// tensor-tensor element-wise ops (grid-stride implementation)
///
__KERN__ void
k_tt_op(math_op op, float *A, float *B, float *O, long n) {
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        switch (op) {                                         /// no divergence
        case ADD: O[j] = A[j] + B[j]; break;
        case SUB: O[j] = A[j] - B[j]; break;
        case MUL: O[j] = A[j] * B[j]; break;                  /// * convolution
        case DIV: O[j] = A[j] / B[j]; break;
        }
    }
}
///
/// tensor-scalar element-wise ops (grid-stride implementation)
///
__KERN__ void
k_ts_op(math_op op, float *A, float v, float *O, long n) {
    for (long j = threadIdx.x; j < n; j += blockDim.x) {
        switch (op) {                                         /// no divergence
        case ADD: O[j] = A[j] + v; break;
        case SUB: O[j] = A[j] - v; break;
        case MUL: O[j] = A[j] * v; break;                     /// * convolution
        case DIV: O[j] = A[j] / v; break;
        }
    }
}
///
/// Binary Cross-Entropy (clamps output to >= -100)
///
__KERN__ void
k_bce(float *O, float *T, long n) {
    for (long j = threadIdx.x; j < n; j+= blockDim.x) {
//        O[i] = ABS(T[i]) < DU_EPS ? LN(DU1 - O[i] + DU_EPS) : LN(O[i] + DU_EPS);
        O[j] = T[j] * LN(O[j] + 1.0e-6) + (1.0f - T[j]) * LN(1.0f - O[j] + 1.0e-6);
    }
}
