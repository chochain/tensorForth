/*! @file
  @brief
  cueForth Asyn IO module

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef CUEF_SRC_AIO_H_
#define CUEF_SRC_AIO_H_
#include "istream.h"
#include "ostream.h"

class AIO : public Managed {
public:
	Istream *_istr;		/// managed input stream
	Ostream *_ostr;		/// managed output stream
	bool    _trace;     /// debug tracing control

	AIO(bool trace) : _istr(new Istream()), _ostr(new Ostream()), _trace(trace) {}

	__HOST__ Istream *istream() { return _istr; }
	__HOST__ Ostream *ostream() { return _ostr; }

	__HOST__ int  readline();
	__HOST__ void flush();

private:
    __HOST__ obuf_node* _print_node(obuf_node *node);
};
#endif // CUEF_SRC_AIO_H_
