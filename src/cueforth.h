/*! @file
  @brief
  cueForth macros and internal class definitions
*/
#ifndef CUEF_SRC_CUEFORTH_H_
#define CUEF_SRC_CUEFORTH_H_
#include "aio.h"

class CueForth {
    AIO  *aio;

public:
    CueForth(bool trace=false);
    ~CueForth();

    __HOST__ int   is_running();
    __HOST__ int   run();
    __HOST__ void  teardown(int sig=0);
};
#endif // CUEF_SRC_CUEFORTH_H_
