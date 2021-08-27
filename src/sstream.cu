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

__GPU__
sstream::sstream(U8 *buf, int sz) : buf(buf), sz(sz) {}

__GPU__ void
sstream::_write(GT gt, U8 *buf, int sz)
{
	if (threadIdx.x!=0) return;						// only thread 0 within a block can write

	_LOCK;

	print_node *n = (print_node *)_output_ptr;
	U8 *d = n->data, *s = buf;
	for (int i=0; i<sz; i++, *d++=*s++);			// mini memcpy

	n->id   = blockIdx.x;							// VM.id
	n->gt   = gt;
	n->size = ALIGN4(sz);							// 32-bit alignment

	_output_ptr  = U8PADD(n->data, n->size);		// advance pointer to next print block
	*_output_ptr = (U8)GT_EMPTY;

	_UNLOCK;
}

__GPU__ U8*
sstream::_va_arg(U8 *p)
{
    U8 ch;
    while ((ch = *p) != '\0') {
        p++;
        if (ch == '%') {
            if (*p == '%') p++;	// is "%%"
            else 	       goto PARSE_FLAG;
        }
    }
    if (ch == '\0') return NULL;

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.precision] type
    //   e.g. "%05d"
    while ((ch = *p)) {
        switch(ch) {
        case '+': case ' ': case '-': case '0': break;
        default : goto PARSE_WIDTH;
        }
        p++;
    }

PARSE_WIDTH:
    int n;
    while ((n = *p - '0'), (0 <= n & n <= 9)) p++;
    if (*p == '.') {
        p++;
        while ((n = *p - '0'), (0 <= n && n <= 9)) p++;
    }
    if (*p) ch = *p++;

    return p;
}

//================================================================
/*! output a character

  @param  c	character
*/
__GPU__ sstream&
sstream::operator<<(U8 c)
{
	char buf[2] = { c, '\0' };
	_write(GT_STR, (U8*)buf, 2);
	return *this;
}

__GPU__ sstream&
sstream::operator<<(GI i)
{
	_write(base==10 ? GT_INT : GT_HEX, (U8*)&i, sizeof(GI));
	return *this;
}

__GPU__ sstream&
sstream::operator<<(GF f)
{
	_write(GT_FLOAT, (U8*)&f, sizeof(GF));
	return *this;
}


//================================================================
/*! output string

  @param str	str
*/
__GPU__ sstream&
sstream::operator<<(const char *str)
{
	U8 *p = (U8*)str;
	int i = 0;	while (*p++ != '\0') i++;	// mini strlen
	_write(GT_STR, (U8*)str, ALIGN4(i+1));
	return *this;
}

__GPU__ sstream&
sstream::operator<<(string s)
{
	this << s.c_str();
	return *this;
}


__GPU__ sstream&
sstream::str(const char *str)
{
	// TODO
	return *this;
}

__GPU__ sstream&
sstream::str(string str)
{
	// TODO
	return *this;
}

__GPU__ sstream& sstream::operator<<(_setbase b) { base  = b.base;  return *this; }
__GPU__ sstream& sstream::operator<<(_setw    w) { width = w.width; return *this; }
__GPU__ sstream& sstream::operator<<(_setfill f) { fill  = f.fill;  return *this; }
__GPU__ sstream& sstream::operator<<(_setprec p) { prec  = p.prec;  return *this; }

__GPU__ sstream&
sstream::operator>>(char **pstr)
{
	// TODO
	return *this;
}

__GPU__ sstream&
sstream::getline(string s, char delim)
{
	// TODO
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


