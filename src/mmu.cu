/** -*- c++ -*-
 * @file
 * @brief tensorForth - Memory Manager
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <iomanip>          // setw, setbase
#include "mmu.h"
///
/// random number generator setup
///
/// assume 256 states for fixed block(16,16)
///
__KERN__ void k_rand_init(curandState *st) {
    int tid = threadIdx.x;
    curand_init(clock64() + tid, tid, 0, &st[tid]);
}
__KERN__ void k_rand(DU *mat, int nrow, int ncol, curandState *st, t4_rand_type n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    curandState *s = &st[threadIdx.x + threadIdx.y * 16];
    
    if (i < ncol && j < nrow) {
        int off = (i + j * ncol);
        mat[off] = n ? curand_normal(s) : curand_uniform(s);  // no divergence
    }
}
///
/// Forth Virtual Machine operational macros to reduce verbosity
///
__HOST__
MMU::MMU() {
    cudaMallocManaged(&_dict, sizeof(Code) * T4_DICT_SZ);
    cudaMallocManaged(&_pmem, T4_PMEM_SZ);
    cudaMallocManaged(&_ten,  T4_TENSOR_SZ);
    cudaMallocManaged(&_mark, sizeof(DU) * T4_TFREE_SZ);
    cudaMallocManaged(&_vss,  sizeof(DU) * T4_SS_SZ * VM_MIN_COUNT);
    cudaMallocManaged(&_seed, sizeof(curandState) * 256);
    GPU_CHK();

    tstore.init(_ten, T4_TENSOR_SZ);
    k_rand_init<<<1, 256>>>(_seed);
    GPU_CHK();
    
    DEBUG("\\  MMU dict=%p, mem=%p, vss=%p, ten=%p\n", _dict, _pmem, _vss, _ten);
}
__HOST__
MMU::~MMU() {
    GPU_SYNC();
    cudaFree(_seed);
    cudaFree(_vss);
    cudaFree(_mark);
    cudaFree(_ten);
    cudaFree(_pmem);
    cudaFree(_dict);
}
///
/// dictionary search functions - can be adapted for ROM+RAM
///
__GPU__ int
MMU::find(const char *s, bool compile, bool ucase) {
    WARN("find(%s) => ", s);
    for (int i = _didx - (compile ? 2 : 1); i >= 0; --i) {
        const char *t = _dict[i].name;
        if (ucase && STRCASECMP(t, s)==0) return i;
        if (!ucase && STRCMP(t, s)==0) return i;
    }
    return -1;
}
///
/// colon - dictionary word compiler
///
__GPU__ void
MMU::colon(const char *name) {
    WARN("colon(%s) => ", name);
    int  sz = STRLENB(name);                // aligned string length
    Code &c = _dict[_didx++];               // get next dictionary slot
    align();                                // nfa 32-bit aligned (adjust _midx)
    c.name = (const char*)&_pmem[_midx];    // assign name field index
    c.def  = 1;                             // specify a colon word
    add((U8*)name,  ALIGN2(sz+1));          // setup raw name field
    c.pfa  = _midx;                         // capture code field index
}
///
/// Debugging methods
///
/// display dictionary word (wastefully one byte at a time)
///
__HOST__ void
MMU::to_s(std::ostream &fout, IU w) {
    /*
     * TODO: not sure why copying 32 byt does not work?
     * char name[36];
     * cudaMemcpy(name, _dict[w].name, 32, D2H);
     */
    U8 c, i=0;
    cudaMemcpy(&c, _dict[w].name, 1, D2H);
    while (c) {
        fout << c;
        cudaMemcpy(&c, _dict[w].name+(++i), 1, D2H);
    }
#if T4_VERBOSE
    fout << " " << w << (_dict[w].immd ? "* " : " ");
#else   // T4_VERBOSE
    fout << " ";
#endif  // T4_VERBOSE
}
///
/// tensor life-cycle methods
///
__GPU__ void
MMU::mark_free(DU v) {            ///< mark a tensor free for release
    Tensor &t = du2ten(v);
    WARN("mark T[%x]=%p as free[%d]\n", *(U32*)&v, &t, _fidx);
//    lock();
    if (_fidx < T4_TFREE_SZ) _mark[_fidx++] = v;
    else ERROR("ERR: tfree array full, increase T4_TFREE_SZ!");
//    unlock();                   ///< TODO: CC: DEAD LOCK, now!
}
__GPU__ void                      ///< release marked free tensor
MMU::sweep() {
//    lock();
    for (int i=0; _fidx && i < _fidx; i++) {
        DU v = _mark[i];
        WARN("release T[%x] from marked list[%d]\n", *(U32*)&v, _fidx);
        free(v);
    }
    _fidx = 0;
//  unlock();                     ///< TODO: CC: DEAD LOCK, now!    
}
__GPU__ Tensor&                    ///< create a one-dimensional tensor
MMU::tensor(U32 sz) {
    Tensor *t    = (Tensor*)tstore.malloc(sizeof(Tensor));
    void   *mptr = tstore.malloc((U64)sizeof(DU) * sz);
    t->reset(mptr, sz);
    return *t;
}
__GPU__ Tensor&                    ///< create a 2-dimensional tensor
MMU::tensor(U16 h, U16 w) {
    U32 sz = h * w;
    DEBUG("mmu#tensor(%d,%d) => size=%d\n", h, w, sz);
    Tensor &t = this->tensor(sz);
    t.reshape(h, w);
    return t;
}
__GPU__ Tensor&                    ///< create a NHWC tensor
MMU::tensor(U16 n, U16 h, U16 w, U16 c) {
    U32 sz = n * h * w * c;
    DEBUG("mmu#tensor(%d,%d,%d,%d) => size=%d\n", n, h, w, c, sz);
    Tensor &t = this->tensor(sz);
    t.reshape(n, h, w, c);
    return t;
}
__GPU__ Tensor&                   ///< create a view of a Tensor
MMU::view(Tensor &t0) {
    Tensor *t = (Tensor*)tstore.malloc(sizeof(Tensor));
    ///
    /// replicate A tensor
    ///
    memcpy(t, &t0, sizeof(Tensor));
    t->attr |= TENSOR_VIEW;
    
    DEBUG("mmu#view:%p => size=%d\n", t, t->size);
    return *t;
}
__GPU__ void                     ///< release tensor memory blocks
MMU::free(Tensor &t) {
    DEBUG("mmu#free(T%d) size=%d\n", t.rank, t.size);
    if (!t.is_view()) tstore.free((void*)t.data);
    tstore.free((void*)&t);
}
__GPU__ Tensor&                  ///< deep copy a tensor
MMU::copy(Tensor &t0) {
    Tensor *t  = (Tensor*)tstore.malloc(sizeof(Tensor));
    memcpy(t, &t0, sizeof(Tensor));
    ///
    /// hard copy data block
    ///
    U64 bsz   = sizeof(DU) * t0.size;
    U8  *mptr = (U8*)tstore.malloc(bsz);
    memcpy(mptr, t0.data, bsz);
    ///
    /// reset attributes
    ///
    t->attr &= ~TENSOR_VIEW;  // not a view
    t->data  = mptr;
    
    return *t;
}

__GPU__ DU
MMU::rand(DU d, t4_rand_type n) {
    if (!IS_TENSOR(d)) return d * curand_uniform(&_seed[0]);
    
    Tensor &t = du2ten(d);
    int h = t.rank==1 ? 1      : t.H();
    int w = t.rank==1 ? t.size : t.W();
    DEBUG("mmu#rand(T%d) size=%d\n", t.rank, t.size);
    dim3 block(16, 16), grid(
        (w + block.x - 1) / block.x,     /* row major */
        (h + block.y - 1) / block.y
        );
    k_rand<<<grid, block>>>((DU*)t.data, h, w, _seed, n);
    return d;
}
///
/// display dictionary word list
///
__HOST__ void
MMU::words(std::ostream &fout) {
    fout << std::setbase(10);
    for (int i=0; i<_didx; i++) {
        if ((i%10)==0) { fout << std::endl; }
        to_s(fout, i);
    }
}
///
/// recursively disassemble colon word
///
__HOST__ void
MMU::see(std::ostream &fout, U8 *ip, int dp) {
    while (*(IU*)ip) {                                              /// * loop until EXIT
        fout << std::endl; for (int n=dp; n>0; n--) fout << "  ";   /// * indentation by level
           fout << "[" << std::setw(4) << (IU)(ip - _pmem) << ": ";
        IU c = *(IU*)ip;                                            /// * fetch word index
        to_s(fout, c);                                              /// * display word name
        if (_dict[c].def && dp < 2) {                               /// * check if is a colon word
            see(fout, &_pmem[_dict[c].pfa], dp+1);                  /// * go one level deeper
        }
        ip += sizeof(IU);                                           /// * advance instruction pointer
        switch (c) {
        case DOVAR: case DOLIT:
            fout << "= " << (*(DU*)ip); ip += sizeof(DU); break;      /// fetch literal
        case DOSTR: case DOTSTR: {
            char *s = (char*)ip;
            int  sz = strlen(s)+1;
            ip += ALIGN2(sz);                                       /// fetch string
            fout << "= \"" << s << "\"";
        } break;
        case BRAN: case ZBRAN: case DONEXT:
            fout << "j" << *(IU*)ip; ip += sizeof(IU); break;       /// fetch jump target
        }
        fout << "] ";
    }
}
__HOST__ void
MMU::see(std::ostream &fout, U16 w) {
    fout << "[ "; to_s(fout, w);
    if (_dict[w].def) see(fout, &_pmem[_dict[w].pfa]);
    fout << "]" << std::endl;
}
///
/// dump data stack content
///
__HOST__ void
MMU::ss_dump(std::ostream &fout, U16 vid, U16 n, int radix) {
    bool x = radix != 10;
    auto show = [this, &fout, x](DU s) {
        if (IS_TENSOR(s)) {
            Tensor &t = this->du2ten(s);
            fout << (char)(t.is_view() ? 'V' : 'T');
            switch(t.rank) {
            case 1: fout << "1[" << t.size << "]"; break;
            case 2: fout << "2[" << t.H() << "," << t.W() << "]"; break;
            case 4: fout << "4[" << t.N() << "," << t.H() << "," << t.W() << "," << t.C() << "]"; break;
            }
        }
        else if (x) fout << static_cast<int>(s);
        else fout << s;
    };
    DU *ss = &_vss[vid * T4_SS_SZ];
    fout << " <";
    if (x) fout << std::setbase(radix);
    for (U16 i=0; i<n; i++) {
        show(ss[i]);
        fout << " ";
    }
    show(ss[T4_SS_SZ-1]);
    fout << "> ok" << std::endl;
}
///
/// Forth pmem memory dump
/// TODO: dynamic parallel
///
#define C2H(c) { buf[x++] = i2h[(c)>>4]; buf[x++] = i2h[(c)&0xf]; }
#define IU2H(i){ C2H((i)>>8); C2H((i)&0xff); }
__HOST__ void
MMU::mem_dump(std::ostream &fout, U16 p0, U16 sz) {
    const char *i2h = "0123456789abcdef";
    char buf[80];
    for (U16 i=ALIGN16(p0); i<=ALIGN16(p0+sz); i+=16) {
        int x = 0;
        buf[x++] = '\n'; IU2H(i); buf[x++] = ':'; buf[x++] = ' ';  // "%04x: "
        for (U16 j=0; j<16; j++) {
            //U8 c = *(((U8*)&_dict[0])+i+j) & 0x7f;               // to dump _dict
            U8 c = _pmem[i+j];
            C2H(c);                                                // "%02x "
            c &= 0x7f;                                             // mask off high bit
            buf[x++] = ' ';
            if (j%4==3) buf[x++] = ' ';
            buf[59+j]= (c==0x7f||c<0x20) ? '.' : c;                // %c
        }
        buf[75] = '\0';
        fout << buf;
    }
    fout << std::endl;
}
