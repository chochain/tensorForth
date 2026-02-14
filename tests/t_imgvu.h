
/**
 * @file
 * @brief tensorForth - Image Viewer class using stb_image.h
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef TEST_IMGVU_H
#define TEST_IMGVU_H
#include "ten4.h"
#include "corpus.h"
#include "vu.h"

class ImgLoader : public Corpus {
public:
    ImgLoader(const char *name) : Corpus(name, NULL) {}
    virtual ImgLoader *load(int bsz=0, int bid=0);
};

class ImgVu : public Vu {
public:
    ImgVu(Corpus &cp) : Vu(cp) {}

    virtual void keyboard(U8 k) { _vuop = (k == '0'); }
    virtual void display(TColor *d_buf) {
        if (_vuop) _img_flip(d_buf);
        else       _img_copy(d_buf);
    }

private:
    int    _vuop = 0;

    void   _img_copy(TColor *d_buf);
    void   _img_flip(TColor *d_buf);
};
#endif // TEST_IMGVU_H
