
/**
 * @file
 * @brief tensorForth - Image Viewer class using stb_image.h
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEST_IMGVU_H
#define TEST_IMGVU_H
#include "../src/vu.h"

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

    int    _load();
    void   _img_copy(TColor *d_dst);
    void   _img_flip(TColor *d_dst);
};
#endif // TEST_BMPVU_H
