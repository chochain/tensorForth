/*! @file
  @brief
  cueForth Asyn IO module

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_AIO_H_
#define CUEF_SRC_AIO_H_
#include "istream.h"
#include "ostream.h"

class AIO {
public:
    static Istream *istream();
    static Ostream *ostream();

    static __HOST__ void init(char *ibuf, char *obuf, bool trace);
	static __HOST__ void flush();
    static __HOST__ obuf_node* _print_node(obuf_node *node);
};
#endif // CUEF_SRC_AIO_H_
