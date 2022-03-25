/*! @file
  @brief
  cueForth macros and internal class definitions
*/
#ifndef CUEF_SRC_CUEFORTH_H_
#define CUEF_SRC_CUEFORTH_H_
#include <sstream>
#include "cuef_config.h"
#include "cuef_types.h"

using namespace std;

class CueForth {
    istream &cin;
    ostream &cout;

    U8 *_heap;
    U8 *_ibuf;
    U8 *_obuf;

    __HOST__ void* _malloc(int sz, int type);
    __HOST__ void  _free(void *mem);

public:
    CueForth(istream &in, ostream &out);
    ~CueForth();

    __HOST__ int   setup(int step=0, int trace=0);
    __HOST__ int   run();
    __HOST__ void  teardown(int sig=0);
};
#endif // CUEF_SRC_CUEFORTH_H_
