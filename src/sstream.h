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
#include "string.h"

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
/// istream class
///
class istream
{
	U8   *buf = NULL;
    int  sz   = 0;
	int  idx  = 0;
    
    __GPU__  U8   *_va_arg(U8 *p);
public:
    __GPU__  istream(U8 *buf, int sz);
    
    // object output
    __GPU__  istream& str(const char *s);
    __GPU__  istream& str(string &s);
    __GPU__  istream& getline(string &s, char delim);
    __GPU__  istream& operator>>(string &s);
};
///
/// iomanip classes
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
/// ostream class
///
class ostream
{
	U8   *buf = NULL;
	int  sz   = 0;
	int  base = 10;
	char fill = ' ';
	int  prec = 6;

    __GPU__  void _write(GT gt, U8 *buf, int sz);
    
public:
    __GPU__  ostream(U8 *buf, int sz);
    
    // object input
    __GPU__ ostream& operator<<(U8 c);
    __GPU__ ostream& operator<<(GI i);
    __GPU__ ostream& operator<<(GF f);
    __GPU__ ostream& operator<<(const char *str);
    __GPU__ ostream& operator<<(string &s);
    // control
    __GPU__ ostream& operator<<(_setbase b);
    __GPU__ ostream& operator<<(_setw    w);
    __GPU__ ostream& operator<<(_setfill f);
    __GPU__ ostream& operator<<(_setprec p);
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
