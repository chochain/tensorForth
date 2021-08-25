/*! @file
  @brief
  GURU console output module. (not yet input)

  <pre>
  Copyright (C) 2019 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/

#ifndef CUEF_SRC_CONSOLE_H_
#define CUEF_SRC_CONSOLE_H_

#include "cuef.h"

//================================================================
/*!@brief
  define the value type.
*/
typedef enum {
    GT_EMPTY = 0,
    GT_INT,
    GT_HEX,
    GT_FLOAT,
    GT_STR,
} GT;

//================================================================
/*! printf internal version data container.
*/
typedef struct {
	U32	id   : 12;
    GT  gt 	 : 4;
    U32	size : 16;
    U8	data[];          								// different from *data
} print_node;

struct print_base {
	int base;
	print_base(int b) : base(b) {}
};

class SStream
{
	int base = 10;

    __GPU__  void _write(GT gt, U8 *buf, int sz);
    __GPU__  U8   *_va_arg(U8 *p);
    
public:
    __GPU__  SStream(U8 *buf, GI sz);

    __GPU__ void operator<<(U8 c);
    __GPU__ void operator<<(GI i);
    __GPU__ void operator<<(GF f);
    __GPU__ void operator<<(const U8 *str);
    __GPU__ void operator<<(print_base &b);
};

// global output buffer for now, per session later
extern __GPU__ GI  _output_size;
extern __GPU__ U8  *_output_buf;
extern __GPU__ U8  *_output_ptr;

__KERN__ void        stream_init(U8 *buf, int sz);
__HOST__ print_node* stream_print(print_node *node, int trace);
__HOST__ void        stream_flush(U8 *output_buf, int trace);

#endif // CUEF_SRC_CONSOLE_H_
