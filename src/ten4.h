/*! @file
  @brief
  tensorForth macros and internal class definitions
*/
#ifndef TEN4_SRC_TEN4_H_
#define TEN4_SRC_TEN4_H_
#include "aio.h"

#define WARP(t)  ((((t) + 32)>>5) << 5)      /** calculate warp thread count */
class TensorForth {
    AIO  *aio;
    MMU  *mmu;
    int  *busy;                 // for Device to Host VM status reporting

public:
    TensorForth(int device=0, int verbose=0);
    ~TensorForth();

    __HOST__ int   is_running();
    __HOST__ int   run();
    __HOST__ void  teardown(int sig=0);
};
#endif // TEN4_SRC_TEN4_H_
