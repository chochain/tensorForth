
/**
 * @file
 * @brief tensorForth - BMP Viewer class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_BMPVU_H
#define TEN4_BMPVU_H
#include "vu.h"

#pragma pack(1)

typedef struct {
    short type;
    int   size;
    short reserved1;
    short reserved2;
    int   offset;
} BMPHeader;

typedef struct {
    int      size;
    int      width;
    int      height;
    short    planes;
    short    bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int      xPelsPerMeter;
    int      yPelsPerMeter;
    int      clrUsed;
    int      clrImportant;
} BMPInfoHeader;

class BmpVu : public Vu {
public:
    BmpVu(const char *fname);

    void   keyboard(U8 k) { _vuop = (k == '0'); }
    void   display(TColor *d_dst) {
        if (_vuop) _img_flip(d_dst);
        else       _img_copy(d_dst);
    }

private:
    int    _vuop = 0;

    void   _load();
    void   _img_copy(TColor *d_dst);
    void   _img_flip(TColor *d_dst);
};
#endif // TEN4_BMPVU_H
