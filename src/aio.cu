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
	std::cin.getline(tib, CUEF_IBUF_SZ, '\n');
	if (_trace) std::cout << "<<" << tib << std::endl;
	return strlen(tib);
}

__HOST__ void
AIO::print_node(obuf_node *node) {
    if (_trace) std::cout << '<' << node->id << '>';

    char *v = (char*)node->data;
    switch (node->gt) {
    case GT_INT:   std::cout << (int)(*(GI*)v);   break;
    case GT_FLOAT: std::cout << (float)(*(GF*)v); break;
    case GT_STR:   std::cout << v;                break;
    case GT_FMT:   {
        obuf_fmt *fmt = (obuf_fmt*)v;
        std::cout << std::setbase(fmt->base)
                  << std::setw(fmt->width)
                  << std::setprecision(fmt->prec)
                  << std::setfill(fmt->fill);
    } break;
    case GT_WORDS: dict->words(); break;
    case GT_SEE:   dict->see();   break;
    case GT_DUMP:  dict->dump();  break;
    default: std::cout << "print type not supported: " << (int)node->gt;  break;
    }
    if (_trace) std::cout << "</" << node->id << '>' << std::endl;
}

#define NEXTNODE(n) ((obuf_node*)((char*)&node->data[0] + node->sz))
__HOST__ void
AIO::flush() {
    obuf_node *node = (obuf_node*)_ostr->rdbuf();
    while (node->gt != GT_EMPTY) {          // 0
        print_node(node);
        node = NEXTNODE(node);
    }
    _ostr->clear();
}


