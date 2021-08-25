/*! @file
  @brief
  cueForth console output module. (not yet input)

  cuef_config.h#CUEF_USE_CONSOLE can switch between CUDA or internal implementation
  <pre>
  Copyright (C) 2021 GreenII

  This file is distributed under BSD 3-Clause License.

  </pre>
*/
#include "cuef.h"
#include "console.h"
#include "sprintf.h"

//================================================================
/*! initialize data container.

  @param  pf	pointer to guru_print
  @param  buf	pointer to output buffer.
  @param  size	buffer size.
  @param  fstr	format string.
*/
__GPU__
SPrinter::SPrinter(U8 *buf, U32 sz)
{
    this->buf = buf;		// point at the start of buf
    this->end = buf + sz;
    this->p   = buf;
}

//================================================================
/*! return string length in buffer

  @param  pf	pointer to guru_print
  @return	length
*/
__GPU__ __INLINE__ int
SPrinter::_size()
{
    return this->end - this->p;
}

//================================================================
/*! terminate ('\0') output buffer.

  @param  pf	pointer to guru_print
*/
__GPU__ __INLINE__ void
SPrinter::_done()
{
    *this->p = '\0';
}

//================================================================
/*! sprintf subcontract function

  @param  pf	pointer to guru_print
  @retval 0		(format string) done.
  @retval 1		found a format identifier.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GPU__ int
SPrinter::_next()
{
    U8  ch = '\0';
    this->fmt = (PrintFormat){0};

    while (_size(pf) && (ch = *this->fstr) != '\0') {
        this->fstr++;
        if (ch == '%') {
            if (*this->fstr == '%') {	// is "%%"
                this->fstr++;
            }
            else goto PARSE_FLAG;
        }
        *this->p++ = ch;
    }
    return -(_size(pf) && ch != '\0');

PARSE_FLAG:
    // parse format - '%' [flag] [width] [.prec] type
    //   e.g. "%05d"
    while ((ch = *(this->fstr))) {
        switch(ch) {
        case '+': this->fmt.plus  = 1; break;
        case ' ': this->fmt.space = 1; break;
        case '-': this->fmt.minus = 1; break;
        case '0': this->fmt.zero  = 1; break;
        default : goto PARSE_WIDTH;
        }
        this->fstr++;
    }

PARSE_WIDTH:
	int n;
    while ((n = *this->fstr - '0'), (0 <= n && n <= 9)) {	// isdigit()
        this->fmt.width = this->fmt.width * 10 + n;
        this->fstr++;
    }
    if (*this->fstr == '.') {
        this->fstr++;
        while ((n = *this->fstr - '0'), (0 <= n && n <= 9)) {
            this->fmt.prec = this->fmt.prec * 10 + n;
            this->fstr++;
        }
    }
    if (*this->fstr) this->fmt.type = *this->fstr++;

    return 1;
}

//================================================================
/*! sprintf subcontract function for U8 '%c'

  @param  pf	pointer to guru_print
  @param  ch	output character (ASCII)
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GPU__ int
SPrinter::_char(U8 ch)
{
    if (this->fmt.minus) {
        if (_size(pf)) *this->p++ = ch;
        else return -1;
    }
    for (int i=0; i < this->fmt.width; i++) {
        if (_size(pf)) *this->p++ = ' ';
        else return -1;
    }
    if (!this->fmt.minus) {
        if (_size(pf)) *this->p++ = ch;
        else return -1;
    }
    return 0;
}

//================================================================
/*! sprintf subcontract function for integer '%d' '%x' '%b'

  @param  pf	pointer to guru_print.
  @param  value	output value.
  @param  base	n base.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GPU__ int
SPrinter::_int(int value, int base)
{
    U32 sign = 0;
    U32 v = value;			// (note) Change this when supporting 64 bit.

    if (this->fmt.type == 'd' || this->fmt.type == 'i') {	// signed.
        if (value < 0) {
            sign = '-';
            v = -value;
        } else if (this->fmt.plus) {
            sign = '+';
        } else if (this->fmt.space) {
            sign = ' ';
        }
    }
    if (this->fmt.minus || this->fmt.width == 0) {
        this->fmt.zero = 0; 		// disable zero padding if left align or width zero.
    }
    this->fmt.prec = 0;

    U32 bias_a = (this->fmt.type == 'X') ? 'A' - 10 : 'a' - 10;

    // create string to local buffer
    U8 buf[64+2];				// int64 + terminate + 1
    U8 *p = buf + sizeof(buf) - 1;
    *p = '\0';
    do {
        U32 i = v % base;
        *--p = (i < 10)? i + '0' : i + bias_a;
        v /= base;
    } while (v != 0);

    // decide pad character and output sign character
    U8 pad;
    if (this->fmt.zero) {
        pad = '0';
        if (sign) {
            *this->p++ = sign;
            if (!_size(pf)) return -1;
            this->fmt.width--;
        }
    }
    else {
        pad = ' ';
        if (sign) *--p = sign;
    }
    return _str((U8*)p, pad);
}

//================================================================
/*! sprintf subcontract function for U8 '%s'

  @param  pf	pointer to guru_print.
  @param  str	output string.
  @param  pad	padding character.
  @retval 0	done.
  @retval -1	buffer full.
  @note		not terminate ('\0') buffer tail.
*/
__GPU__ int
SPrinter::_str(U8 *str, U8 pad)
{
	U32 len = STRLENB(str);
    S32 ret = 0;

    if (str == NULL) {
        str = (U8*)"(null)";
        len = 6;
    }
    if (this->fmt.prec && len > this->fmt.prec) len = this->fmt.prec;

    S32 tw = len;
    if (this->fmt.width > len) tw = this->fmt.width;

    ASSERT(len <= _size(pf));
    ASSERT(tw  <= _size(pf));

    S32 n_pad = tw - len;
    S32 minus = this->fmt.minus;

    if (!minus) {										// left padding
    	MEMSET(this->p, pad, n_pad);	this->p += n_pad;
    }
    MEMCPY(this->p, str, len);	this->p += len;
    if (minus) {
    	MEMSET(this->p, pad, n_pad);	this->p += n_pad;		// right padding
    }
    return ret;
}

//================================================================
/*! sprintf subcontract function for float(double) '%f'

  @param  pf	pointer to guru_print.
  @param  value	output value.
  @retval 0	done.
  @retval -1	buffer full.
*/
__GPU__ int
SPrinter::_float(double v)
{
    U8  fstr[16];
    U8 *p0 = (U8*)this->fstr;
    U8 *p1 = fstr + sizeof(fstr) - 1;

    *p1 = '\0';
    while ((*--p1 = *--p0) != '%');

    // TODO: 20181025 format print float
    //snprintf(this->p, (this->buf_end - this->p + 1), p1, value);

    while (*this->p != '\0') this->p++;

    return _size(pf);
}

//================================================================
/*! output formatted string

  @param  buf       output buffer
  @param  fstr		format string.
*/
__GPU__ void
SPrinter::print(const U8 *fstr, ...)
{
    this->fmt = (PrintFormat){0};
    this->fstr= fstr;
    
    va_list ap;
    va_start(ap, fstr);
    
    U32 x = 0;
    while (x==0 && _next()) {
    	switch(this->fmt.type) {
        case 'c': x = _char(va_arg(ap, int));        	 	break;
        case 's': x = _str(va_arg(ap, U8 *), ' '); 	 		break;
        case 'd':
        case 'i':
        case 'u': x = _int(va_arg(ap, unsigned int), 10); 	break;
        case 'b':
        case 'B': x = _int(va_arg(ap, unsigned int), 2);  	break;
        case 'x':
        case 'X': x = _int(va_arg(ap, unsigned int), 16); 	break;
        case 'f':
        case 'e':
        case 'E':
        case 'g':
        case 'G': x = _float(va_arg(ap, double)); 		 	break;
        default: x=0;
        }
    }
    va_end(ap);
    _done();
}

SStream::SStream() : sp(new SPrinter(buf, CUEF_CONSOLE_BUF_SIZE) {}
SStream::operator<<(U8 ch)              { sp.print("%c", ch); }
SStream::operator<<(int i, int base=10) { sp.print((base==10 ? "%d" : "%x"), i); }
SStream::operator<<(U8 *str)            { sp.print("%s", str); }
SStream::operator<<(float f)            { sp.print("%f", (float)f); }


