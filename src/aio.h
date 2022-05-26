/*! @file
  @brief
  tensorForth Asyn IO module

  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#ifndef TEN4_SRC_AIO_H_
#define TEN4_SRC_AIO_H_
#include "istream.h"
#include "ostream.h"
#include "mmu.h"

class AIO : public Managed {
public:
    Istream *_istr;         /// managed input stream
    Ostream *_ostr;         /// managed output stream
    MMU     *_mmu;          /// memory managing unit
    bool    _trace;         /// debug tracing control

    AIO(MMU *mmu, bool trace) : _istr(new Istream()), _ostr(new Ostream()), _mmu(mmu), _trace(trace) {}

    __HOST__ Istream *istream() { return _istr; }
    __HOST__ Ostream *ostream() { return _ostr; }

    __HOST__ int  readline();
    __HOST__ void print_node(obuf_node *node);
    __HOST__ void flush();
};
#endif // TEN4_SRC_AIO_H_
