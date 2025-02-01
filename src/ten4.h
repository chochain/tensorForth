/*! 
  @file
  @brief TensorForth class - macros and internal class definitions
*/
#ifndef TEN4_SRC_TEN4_H_
#define TEN4_SRC_TEN4_H_
#include "vm/vm.h"           // proxy to sys.h

#if T4_ENABLE_NN
#include "ldr/loader.h"      // default dataset loader
#include "vm/netvm.h"        // neural network set,
typedef NetVM   VM_TYPE;
#elif T4_ENABLE_OBJ
#include "vm/tenvm.h"        // tensor/matrix set, or
typedef TenVM   VM_TYPE;
#else
#include "vm/eforth.h"       // just eForth
typedef ForthVM VM_TYPE;
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
    __HOST__ int   run();                    ///< execute tensorForth main loop
    __HOST__ int   tally();                  ///< tally fetch state of VMs
    __HOST__ void  teardown(int sig=0);
    
private:    
    __HOST__ void  _once();
    __HOST__ void  _profile();
};
#endif // TEN4_SRC_TEN4_H_
