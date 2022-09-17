/*! @file
  @brief
  tensorForth macros and internal class definitions
*/
#ifndef TEN4_SRC_TEN4_H_
#define TEN4_SRC_TEN4_H_
#include "vm.h"

#define WARP(t)  ((((t) + 32)>>5) << 5)   /** calculate warp thread count */
class TensorForth {
    AIO  *aio;                            ///< async IO manager
    MMU  *mmu;                            ///< memory management unit
    int  *busy;                           ///< for Device to Host VM status reporting

public:
    TensorForth(int device=0, int verbose=0);
    ~TensorForth();

    __HOST__ int   is_ready();           ///< has worker ready to take input
    __HOST__ int   run();                ///< execute tensorForth main loop
    __HOST__ void  teardown(int sig=0);
};
#endif // TEN4_SRC_TEN4_H_
