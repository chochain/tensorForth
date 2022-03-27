/*! @file
  @brief
  cueForth macros and internal class definitions
*/
#ifndef CUEF_SRC_CUEFORTH_H_
#define CUEF_SRC_CUEFORTH_H_
#include "cuef_config.h"
#include "cuef_types.h"

class CueForth {
	U8   *_heap;
    U8   *_ibuf;
    U8   *_obuf;
    AIO  *aio;

    __HOST__ void* _malloc(int sz, int type);
    __HOST__ void  _free(void *mem);

public:
    CueForth();
    ~CueForth();

    __HOST__ int   setup(int step=0, int trace=0);
    __HOST__ int   is_running();
    __HOST__ int   run();
    __HOST__ void  teardown(int sig=0);
};
#endif // CUEF_SRC_CUEFORTH_H_
