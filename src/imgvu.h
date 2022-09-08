
/**
 * @file
 * @brief tensorForth - Image Viewer class
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEN4_IMGVU_H
#define TEN4_IMGVU_H

#include "vu.h"

typedef U32 TColor;

extern "C" int  gui_init(int *argc, char **argv, Vu *vu, int x, int y);
extern "C" int  gui_loop();

extern "C" void load_bmp(const char *fname, uchar4 **bmp, int *h, int *w);

class ImgVu : public Vu {
public:
    ImgVu(const char *fname);
    
    void   keyboard(U8 k) { _vuop = (k == '0'); }
    void   display(TColor *d_dst) {
        if (_vuop) _img_flip(d_dst);
        else       _img_copy(d_dst);
    }

private:
    int    _vuop = 0;

    void   _img_copy(TColor *d_dst);
    void   _img_flip(TColor *d_dst);
};
#endif // TEN4_IMGVU_H
