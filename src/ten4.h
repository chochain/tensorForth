/*! 
  @file
  @brief TensorForth class - macros and internal class definitions
*/
#ifndef TEN4_SRC_TEN4_H_
#define TEN4_SRC_TEN4_H_
#include <curand_kernel.h>
#include "vm/vm.h"

#if T4_ENABLE_NN
#include "ldr/loader.h"      // default dataset loader
#include "vm/netvm.h"        // neural network set,
#elif T4_ENABLE_OBJ
#include "vm/tenvm.h"        // tensor/matrix set, or
#else
#include "vm/eforth.h"       // just eForth
#endif

#define WARP(t)  ((((t) + 32)>>5) << 5)   /** calculate warp thread count */
#define VMST_SZ  (sizeof(vm_state) * VM_MIN_COUNT)
#define VMSS_SZ  (sizeof(DU) * T4_SS_SZ * VM_MIN_COUNT)

typedef enum {
    UNIFORM = 0,
    NORMAL
} t4_rand_opt;

class TensorForth {
public:    
    AIO            *aio;            ///< async IO manager
    MMU            *mmu;            ///< memory management unit
    vm_state       *vmst;           ///< VM state on Managed memory
    int            *vmst_cnt;       ///< state tally
private:
    int            _khz;            ///< GPU clock speed
    int            _trace = 0;      ///< debug tracing verbosity level
    curandState    *_seed;          ///< for random number generator

public:
    TensorForth(int device=0, int verbose=0);
    ~TensorForth();

    __HOST__ int   vm_tally();            ///< tally fetch state of VMs
    __HOST__ int   run();                 ///< execute tensorForth main loop
    __HOST__ void  teardown(int sig=0);
    ///
    /// debugging methods (implemented in .cu)
    ///
    __GPU__  __INLINE__ DU   ms()           { return static_cast<double>(clock64()) / _khz; }
    __BOTH__ __INLINE__ int  khz()          { return _khz;   }
    __BOTH__ __INLINE__ int  trace()        { return _trace; }
    __BOTH__ __INLINE__ void trace(int lvl) { _trace = lvl;  }
};
#endif // TEN4_SRC_TEN4_H_
