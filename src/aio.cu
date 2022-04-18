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
	if (_trace) printf("<< %s\n", tib);
	return strlen(tib);
}

__HOST__ void
AIO::print_node(obuf_node *node) {
    if (_trace) printf("<%d>", node->id);

    char *v = (char*)node->data;
    switch (node->gt) {
    case GT_INT:   printf("%d", *(GI*)v); break;
    case GT_HEX:   printf("%x", *(GI*)v); break;
    case GT_FLOAT: printf("%G", *(GF*)v); break;
    case GT_STR:   printf("%s", v);       break;
    default: printf("print node type not supported: %d", node->gt); break;
    }
    if (_trace) printf("</%d>\n", node->id);
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


