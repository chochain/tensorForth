/*! @file
  @brief
  cueForth macros and internal class definitions
*/
#ifndef CUEF_SRC_CUEFORTH_H_
#define CUEF_SRC_CUEFORTH_H_
#include <sstream>
#include "cuef_config.h"
#include "cuef.h"

using namespace std;

#define PRINTF				printf
#define NA(msg)				({ PRINTF("method not supported: %s\n", msg); })

class CueForth {
	istream &cin;
	ostream &cout;

	U8 *heap;
	U8 *ibuf;
	U8 *obuf;

    __HOST__ void* _malloc(int sz, int type);
    __HOST__ void  _free(void *mem);

public:
    CueForth(istream &in, ostream &out);
    __HOST__ int   setup(int step=0, int trace=0);
    __HOST__ int   run();
    __HOST__ void  teardown(int sig=0);
};
#endif // CUEF_SRC_CUEFORTH_H_
