/*! @file
  @brief
  cueForth stream output module. (not yet input)

  <pre>
  Copyright (C) 2019- GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "sstream.h"

#define _LOCK		{ MUTEX_LOCK(_mutex_ss); }
#define _UNLOCK		{ MUTEX_FREE(_mutex_ss); }

namespace cuef {

__GPU__ volatile int _mutex_ss;
///
/// ostream output buffer writer
///
__GPU__ void
ostream::_write(GT gt, U8 *v, int sz)
{
	if (threadIdx.x!=0) return;						// only thread 0 within a block can write

	_LOCK;

	print_node *n = (print_node *)_output_ptr;
	MEMCPY(n->data, v, sz);

	n->id   = blockIdx.x;							// VM.id
	n->gt   = gt;
	n->size = ALIGN4(sz);							// 32-bit alignment

	_output_ptr  = U8PADD(n->data, n->size);		// advance pointer to next print block
	*_output_ptr = (U8)GT_EMPTY;

	_UNLOCK;
}

} // namespace cuef

__GPU__ int _output_size;
__GPU__ U8  *_output_buf;
__GPU__ U8  *_output_ptr;		// global output buffer for now, per session later

__KERN__ void
stream_init(U8 *buf, int sz)
{
	if (threadIdx.x!=0 || blockIdx.x!=0) return;

	print_node *node = (print_node *)buf;
	node->gt = GT_EMPTY;

	_output_buf = _output_ptr = buf;

	if (sz) _output_size = sz;					// set to new size
}

#define NEXTNODE(n)	((print_node *)(node->data + node->size))

__HOST__ print_node*
stream_print(print_node *node, int trace)
{
	U8 buf[80];									// check buffer overflow

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
stream_flush(U8 *output_buf, int trace)
{
	GPU_SYNC();
	print_node *node = (print_node *)output_buf;
	while (node->gt != GT_EMPTY) {			// 0
		node = stream_print(node, trace);
		node = NEXTNODE(node);
	}
	stream_init<<<1,1>>>(output_buf, 0);
	GPU_SYNC();
}


