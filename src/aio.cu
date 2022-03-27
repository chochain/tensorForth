/*! @file
  @brief
  cueForth Async IO module implementation

  <pre>
  Copyright (C) 2021- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include <cstdio>
#include "aio.h"

__GPU__ Istream *istr;
__GPU__ Ostream *ostr;

__KERN__ void
_aio_setup(char *ibuf, char *obuf) {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;

    istr = new Istream(ibuf);
    ostr = new Ostream(obuf);
}

__KERN__ void
_aio_reset() {
    if (threadIdx.x!=0 || blockIdx.x!=0) return;

    ostr->clear();
}

#define NEXTNODE(n) ((obuf_node *)(node->data + node->size))
///
/// AIO takes managed memory blocks as input and output buffers
/// which can be access by both device and host
///
AIO::AIO(char *ibuf, char *obuf) : _ibuf(ibuf), _obuf(obuf) {
    _aio_setup<<<1,1>>>(ibuf, obuf);
    trace = 1;
}

__HOST__ Istream*
AIO::istream() { return istr; }

__HOST__ Ostream*
AIO::ostream() { return ostr; }

__HOST__ obuf_node*
AIO::_print_node(obuf_node *node) {
    U8 buf[80];                                 // check buffer overflow

    if (trace) printf("<%d>", node->id);

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
    if (trace) printf("</%d>\n", node->id);

    return node;
}

__HOST__ void
AIO::flush() {
    obuf_node *node = (obuf_node *)_obuf;
    while (node->gt != GT_EMPTY) {          // 0
        node = _print_node(node);
        node = NEXTNODE(node);
    }
    _aio_reset<<<1,1>>>();
}
