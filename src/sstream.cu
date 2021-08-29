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
///
/// istream class
///
__GPU__ volatile int _mutex_ss;

__GPU__
istream::istream(U8 *buf) : buf(buf) {}

__GPU__ istream&
istream::str(const char *s, int sz)
{
	MEMCPY(buf, s, sz ? sz : STRLENB(s));
	idx = 0;
	return *this;
}

__GPU__ istream&
istream::str(string& s)
{
	str(s.c_str(), s.size());
	return *this;
}

__GPU__ int
istream::getline(string& s, char delim)
{
    while (delim==' ' &&
           (buf[idx]==' ' || buf[idx]=='\t')) idx++; // skip leading blanks and tabs
    int i = idx;
    while (buf[i] && buf[i]!=delim) i++;
    if (buf[i] != delim) return 0;
    s.n = 0;
    s.merge((char*)&buf[idx], i-idx);
    idx = i;
    return s.n;
}

__GPU__ int
istream::operator>>(string& s)
{
	return getline(s);
}
///
/// ostream class
///
__GPU__
ostream::ostream(U8 *buf, int sz) : buf(buf), sz(sz) {}
///
/// output buffer writer
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
///
/// iomanip classes
///
__GPU__ ostream& ostream::operator<<(_setbase b) { base  = b.base;  return *this; }
__GPU__ ostream& ostream::operator<<(_setw    w) { width = w.width; return *this; }
__GPU__ ostream& ostream::operator<<(_setfill f) { fill  = f.fill;  return *this; }
__GPU__ ostream& ostream::operator<<(_setprec p) { prec  = p.prec;  return *this; }
//================================================================
/*! output a character

  @param  c	character
*/
__GPU__ ostream&
ostream::operator<<(U8 c)
{
	U8 buf[2] = { c, '\0' };
	_write(GT_STR, buf, 2);
	return *this;
}

__GPU__ ostream&
ostream::operator<<(GI i)
{
	_write(base==10 ? GT_INT : GT_HEX, (U8*)&i, sizeof(GI));
	return *this;
}

__GPU__ ostream&
ostream::operator<<(GF f)
{
	_write(GT_FLOAT, (U8*)&f, sizeof(GF));
	return *this;
}

//================================================================
/*! output string

  @param str	str
*/
__GPU__ ostream&
ostream::operator<<(const char *s)
{
	int i = STRLENB(s);
	_write(GT_STR, (U8*)s, i);
	return *this;
}

__GPU__ ostream&
ostream::operator<<(string &s)
{
	U8 *s1 = (U8*)s.c_str();
	int i  = STRLENB(s1);
	_write(GT_STR, s1, i);
	return *this;
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


