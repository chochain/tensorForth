/*! 
  @file
  @brief TensorForth class - macros and internal class definitions
*/
#ifndef __TEN4_H_
#define __TEN4_H_
#pragma once
#include "vm/vm.h"
#include "sys.h"

namespace t4::vm {
VM* vm_factory(vm::vm_level level, int id, System &sys);  /// * extern (in vm/vm.cpp)
}

namespace t4 {
using vm::VM;

struct VM_Handle {
    VM      *vm;                             ///< polymorphic handle
    STREAM  st;                              ///< CUDA stream
    EVENT   t0;                              ///< CUDA starting event
    EVENT   t1;                              ///< CUDA end event
};

class TensorForth {
    System    *sys;
    VM_Handle vm_pool[T4_VM_COUNT];          ///< VM handles
    int       vmst_cnt[vm::VM_STATE_MAX];    ///< state counters
    
public:
    TensorForth(int device=0, int verbose=0);

    __HOST__ void  setup(const char *tb_logdir=NIL, const char *tb_run_id=NIL);
    __HOST__ int   more_job();               ///< tally fetch state of VMs
    __HOST__ void  run();                    ///< run (and profile) VMs once
    __HOST__ void  profile();                ///< profile VM elapse
    __HOST__ int   main_loop();              ///< execute tensorForth main loop
    __HOST__ void  teardown(int sig=0);
};

} // namespace t4
#endif // __TEN4_H_
