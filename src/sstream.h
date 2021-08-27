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
#include "cuef.h"
#include "vector.h"

//================================================================
/*! printf internal version data container.
*/
typedef struct {
	U32	id   : 12;
    U32 gt 	 : 4;
    U32	size : 16;
    U8	data[];          								// different from *data
} print_node;

namespace cuef {
///
/// stringbuf class
///
struct string : public vector<char>
{
	__GPU__ string(int asz=16) {
		n = 0; if (asz>0) v = (char*)malloc(sz=asz);
	}
	__GPU__ string(char *s, int asz=16) {
		n  = STRLENB(s);
		sz = ALIGN4(asz>n ? asz : n);
		v  = (char*)malloc(sz);
		MEMCPY(v, s, n);
	}
	__GPU__ ~string() { if (v) free(v); }

	__GPU__ string& operator<<(string s)     { merge(s.v, s.size());        return *this; }
	__GPU__ string& operator<<(const char *s){ merge((char*)s, STRLENB(s)); return *this; }
	__GPU__ bool    operator==(string s)     { return STRCMP(v, s.v); }

	__GPU__ char *c_str() { return v; }
};
    
//================================================================
/*!@brief
  define the value type.
*/
typedef enum {
    GT_EMPTY = 0,
    GT_INT,
    GT_HEX,
    GT_FLOAT,
    GT_STR,
} GT;

///
/// iomanip
///
class _setbase { public: int  base;  __GPU__ _setbase(int b) : base(b)  {}};
class _setw    { public: int  width; __GPU__ _setw(int w)    : width(w) {}};
class _setfill { public: char fill;  __GPU__ _setfill(char f): fill(f)  {}};
class _setprec { public: int  prec;  __GPU__ _setprec(int p) : prec(p)  {}};
__GPU__ __INLINE__ _setbase setbase(int b)  { return _setbase(b); }
__GPU__ __INLINE__ _setw    setw(int w)     { return _setw(w);    }
__GPU__ __INLINE__ _setfill setfill(char f) { return _setfill(f); }
__GPU__ __INLINE__ _setprec setprec(int p)  { return _setprec(p); }
///
/// sstream class
///
class sstream
{

	U8   *buf;
	int  sz   = 0;
	int  base = 10;
	char fill = ' ';
	int  prec = 6;

    __GPU__  void _write(GT gt, U8 *buf, int sz);
    __GPU__  U8   *_va_arg(U8 *p);
    
public:
    __GPU__  sstream(U8 *buf, int sz);

    __GPU__ sstream& operator<<(U8 c);
    __GPU__ sstream& operator<<(GI i);
    __GPU__ sstream& operator<<(GF f);
    __GPU__ sstream& operator<<(const char *str);
    __GPU__ sstream& operator<<(string s);

    __GPU__ sstream& operator<<(_setbase b);
    __GPU__ sstream& operator<<(_setw    w);
    __GPU__ sstream& operator<<(_setfill f);
    __GPU__ sstream& operator<<(_setprec p);

    __GPU__ sstream& str(const char *s);
    __GPU__ sstream& str(string s);
    __GPU__ sstream& operator>>(char **str);
    __GPU__ sstream& getline(string s, char delim);
};

}   // namespace cuef

// global output buffer for now, per session later
extern __GPU__ GI  _output_size;
extern __GPU__ U8  *_output_buf;
extern __GPU__ U8  *_output_ptr;

__KERN__ void        stream_init(U8 *buf, int sz);
__HOST__ print_node* stream_print(print_node *node, int trace);
__HOST__ void        stream_flush(U8 *output_buf, int trace);

#endif // CUEF_SRC_SSTREAM_H_
