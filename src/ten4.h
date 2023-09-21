/*! 
  @file
  @brief TensorForth class - macros and internal class definitions
*/
#ifndef TEN4_SRC_TEN4_H_
#define TEN4_SRC_TEN4_H_
#include "vm.h"

#if T4_ENABLE_OBJ
#include "tenvm.h"        // tensor/matrix set, or
#else
#include "eforth.h"       // just eForth
#endif

#define WARP(t)  ((((t) + 32)>>5) << 5)   /** calculate warp thread count */
#define VMST_SZ  (sizeof(vm_state) * VM_MIN_COUNT)
#define VMSS_SZ  (sizeof(DU) * T4_SS_SZ * VM_MIN_COUNT)

class TensorForth {
    AIO      *aio;                        ///< async IO manager
    MMU      *mmu;                        ///< memory management unit
    vm_state *vmst;                       ///< VM state on Managed memory
    int      *vmst_cnt;                   ///< state tally

public:
    TensorForth(int device=0, int verbose=0);
    ~TensorForth();

    __HOST__ int   vm_tally();            ///< tally fetch state of VMs
    __HOST__ int   run();                 ///< execute tensorForth main loop
    __HOST__ void  teardown(int sig=0);
};
#endif // TEN4_SRC_TEN4_H_
