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
	printf("<< %s\n", tib);
	return strlen(tib);
}

#define NEXTNODE(n) ((obuf_node *)(node->data + node->size))
__HOST__ void
AIO::flush() {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        node = _print_node(node);
        node = NEXTNODE(node);
    }
    _ostr->clear();
}

__HOST__ obuf_node*
AIO::_print_node(obuf_node *node) {
    U8 buf[80];                                 // check buffer overflow

    if (_trace) printf("<%d>", node->id);

    switch (node->gt) {
    case GT_INT:
        printf("%d", *((GI*)node->data));
        break;
    case GT_HEX:
        printf("%x", *((GI*)node->data));
        break;
    case GT_FLOAT:
        printf("%g", *((GF*)node->data));
        break;
    case GT_STR:
        memcpy(buf, (U8*)node->data, node->size);
        printf("%s", (U8*)buf);
        break;
    default: printf("print node type not supported: %d", node->gt); break;
    }
    if (_trace) printf("</%d>\n", node->id);

    return node;
}

