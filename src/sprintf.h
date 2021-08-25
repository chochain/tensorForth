/*! @file
  @brief
  cueForth console output module. (not yet input)

  cuef_config.h#CUEF_USE_CONSOLE can switch between CUDA or internal implementation

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_SPRINTF_H_
#define CUEF_SRC_SPRINTF_H_
#include <cstdio>
#include <cstdarg>
#include "cuef.h"

//================================================================
/*! printf internal version data container.
 */
typedef struct sPrintFormat {
	U32 			type  : 8;			//!< format char. (e.g. 'd','f','x'...)
    U32 			plus  : 1;
    U32 			minus : 1;
    U32 			space : 1;
    U32 			zero  : 1;
    U32 			prec  : 4;		    //!< precision (e.g. %5.2f as 2)
    U32 			width : 16;			//!< display width. (e.g. %10d as 10)
} PrintFormat;

class SPrinter {
    PrintFormat 	fmt;
    const U8 		*fstr;				//!< format string. (e.g. "%d %03x")
    const U8       	*buf;				//!< output buffer.
    const U8		*end;				//!< output buffer end point.
    U8       		*p;					//!< output buffer write point.

    __GPU__ int  _size();
    __GPU__ void _done();
    __GPU__ int  _next();
    __GPU__ int  _char(U8 ch);
    __GPU__ int  _int(int value, int base=10);
    __GPU__ int  _str(U8 *str, U8 pad);
    __GPU__ int  _float(double value);
    
public:    
    __GPU__ SPrinter(U8 *buf, U32 sz);
    __GPU__ void print(const U8 *fstr, ...);			// fstr is always static string (char *)
}
#endif // CUEF_SRC_SPRINTF_H_
