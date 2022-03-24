/*! @file
  @brief
  cueForth string stream module.

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_SSTREAM_H_
#define CUEF_SRC_SSTREAM_H_
#include "istream.h"
#include "ostream.h"

// global output buffer for now, per session later
extern __GPU__ GI  _output_size;
extern __GPU__ U8  *_output_buf;
extern __GPU__ U8  *_output_ptr;

__KERN__ void        stream_init(U8 *buf, int sz);
__HOST__ obuf_node*  stream_print(obuf_node *node, int trace);
__HOST__ void        stream_flush(U8 *output_buf, int trace);

#endif // CUEF_SRC_SSTREAM_H_
