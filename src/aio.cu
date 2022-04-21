/*! @file
  @brief
  cueForth Async IO module implementation

  <pre>
  Copyright (C) 2021- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <cstdio>        // printf
#include <iostream>      // cin, cout
#include <iomanip>       // setbase, setprecision
#include "aio.h"
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
__HOST__ int
AIO::readline() {
	_istr->clear();
	char *tib = _istr->rdbuf();
	std::cin.getline(tib, CUEF_IBUF_SIZE, '\n');
	if (_trace) std::cout << "<<" << tib << std::endl;
	return strlen(tib);
}

__HOST__ void
AIO::print_node(obuf_node *node) {
    if (_trace) std::cout << '<' << node->id << '>';

    char *v = (char*)node->data;
    switch (node->gt) {
    case GT_INT:   std::cout << std::setw(2) << (int)(*(GI*)v);           break;
    case GT_HEX:   std::cout << std::setbase(16) << (int)(*(GI*)v);       break;
    case GT_FLOAT: std::cout << std::setprecision(6) << (float)(*(GF*)v); break;
    case GT_STR:   std::cout << v;                                        break;
    default: std::cout << "print type not supported: " << (int)node->gt;  break;
    }
    if (_trace) std::cout << "</" << node->id << '>' << std::endl;
}

#define NEXTNODE(n) ((obuf_node*)((char*)&node->data[0] + node->size))
__HOST__ void
AIO::flush() {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        print_node(node);
        node = NEXTNODE(node);
    }
    _ostr->clear();
}


