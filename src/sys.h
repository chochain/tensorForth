/** 
 * @file
 * @brief System class - tensorForth System interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_SRC_SYS_H
#define TEN4_SRC_SYS_H
#include <curand_kernel.h>
#include "ten4_types.h"
#include "mmu/mmu.h"
#include "aio/aio.h"
#include "ldr/loader.h"

typedef enum { RDX=0, CR, DOT, UDOT, EMIT, SPCS } io_op;

class System : public Managed {                 ///< singleton class
    AIO            *aio;                        ///< async IO manager
    MMU            *mmu;                        ///< memory management unit
    
    int            _khz;                        ///< GPU clock speed
    int            _trace;                      ///< debug tracing verbosity level
    curandState    *_seed;                      ///< for random number generator
    
public:
    __HOST__ System(int khz, int verbose=0)
        : _khz(khz), _trace(verbose) {
        mmu = new MMU();                        ///> instantiate memory manager
        aio = new AIO(mmu);                     ///> instantiate async IO manager
        
#if (T4_ENABLE_OBJ && T4_ENABLE_NN)
        Loader::init(verbose);
#endif
    }
    __HOST__ ~System() {
        delete aio;
        cudaDeviceReset();
    }
    ///
    /// System functions
    ///
    __GPU__  __INLINE__ DU   ms()           { return static_cast<double>(clock64()) / _khz; }
    __GPU__  DU   rand(DU d, t4_rand_opt n);              ///< randomize a tensor
    ///
    /// debugging methods (implemented in .cu)
    ///
    __BOTH__ __INLINE__ int  khz()          { return _khz;   }
    __BOTH__ __INLINE__ int  trace()        { return _trace; }
    __BOTH__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
    
    __HOST__ void fin_setup(const char *line);
    __HOST__ void fout_setup(void (*hook)(int, const char*));
    ///
    ///> IO functions
    ///
    __GPU__  char *scan(char c);                          ///< scan input stream for a given char
    __GPU__  int  fetch(string &idiom);                   ///< read input stream into string
    __GPU__  char *word();                                ///< get next idiom
    
    __GPU__  char key();                                  ///< read key from console
    __GPU__  void load(VM &vm, const char* fn);           ///< load external Forth script
    __GPU__  void spaces(int n);                          ///< show spaces
    __GPU__  void dot(io_op op, DU v=DU0);                ///< print literals
    __GPU__  void dotr(int w, DU v, int b, bool u=false); ///< print fixed width literals
    __GPU__  void pstr(const char *str, io_op op=SPCS);   ///< print string
    ///
    ///> Debug functions
    ///
#if T4_ENABLE_OBJ // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    __HOST__ int  to_s(std::ostream &fout, DU s);         ///< dump object from descriptor
    __HOST__ int  to_s(std::ostream &fout, T4Base &t, bool view); ///< dump object on stack
#endif    
    __HOST__ void ss_dump(VM &vm, bool forced=false);     ///< show data stack content
    __HOST__ void see(IU pfa, int base);                  ///< disassemble user defined word
    __HOST__ void words(int base);                        ///< list dictionary words
    __HOST__ void dict_dump(int base);                    ///< dump dictionary
    __HOST__ void mem_dump(U32 addr, IU sz, int base);    ///< dump memory frm addr...addr+sz
    __HOST__ void mem_stat();                             ///< display memory statistics
};
#endif // TEN4_SRC_SYS_H

