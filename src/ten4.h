/*! 
  @file
  @brief TensorForth class - macros and internal class definitions
*/
#ifndef __TEN4_H_
#define __TEN4_H_
#include "vm/vm.h"           // proxy to sys.h
#include "vm/eforth.h"       // just eForth
#include "vm/tenvm.h"        // tensor/matrix set, or
#include "vm/netvm.h"        // neural network set,

#if T4_DO_NN
typedef NetVM     VM_TYPE;
#elif T4_DO_OBJ
typedef TensorVM  VM_TYPE;
#else
typedef ForthVM   VM_TYPE;
#endif

#define WARP_SZ   32                        /** threads per warp       */
#define WARP(t)   ((((t) + 31)>>5) << 5)    /** calculate block number */

struct VM_Handle : public Managed {
    VM_TYPE *vm;
    STREAM  st;
    EVENT   t0;
    EVENT   t1;
};

class TensorForth {
    System    *sys;
    VM_Handle *vm_pool;                      ///< CUDA stream per VM
    int       *vmst_cnt;
    
public:
    TensorForth(int device=0, int verbose=0);
    ~TensorForth();

    __HOST__ void  setup();
    __HOST__ int   more_job();               ///< tally fetch state of VMs
    __HOST__ void  run();                    ///< run (and profile) VMs once
    __HOST__ void  profile();                ///< profile VM elapse
    __HOST__ int   main_loop();              ///< execute tensorForth main loop
    __HOST__ void  teardown(int sig=0);
};
#endif // __TEN4_H_
