
/**
 * @file
 * @brief tensorForth - BMP Viewer class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEST_BMPVU_H
#define TEST_BMPVU_H
#include "corpus.h"
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

class BmpLoader : public Corpus {
public:
    BmpLoader(const char *name) : Corpus(name, NULL, 0) {}
    virtual BmpLoader *load(int bsz=0, int bid=0);
};

class BmpVu : public Vu {
public:
    BmpVu(Corpus &cp) : Vu(cp) {}
    
    virtual void   keyboard(U8 k) { _vuop = (k == '0'); }
    virtual void   display(TColor *d_buf) {
        if (_vuop) _img_flip(d_buf);
        else       _img_copy(d_buf);
    }

private:
    int    _vuop = 0;

    void   _img_copy(TColor *d_buf);
    void   _img_flip(TColor *d_buf);
};
#endif // TEST_BMPVU_H
