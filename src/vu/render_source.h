/**
 * @file
 * @brief IRenderSource - the contract between the CUDA-side data model
 *        (Vu) and the OpenGL-side viewer (gui). gui.cu only ever talks to
 *        objects through this interface, so it never needs to know that
 *        Vu, Corpus, or any CUDA type exists.
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VU_RENDER_SOURCE_H
#define __VU_RENDER_SOURCE_H
#pragma once
#include "ten4_types.h"        /// * U8, U32, __HOST__ - no GL, no CUDA runtime

namespace t4::vu {

typedef U32 TColor;

/**
 * @brief Anything that can be shown in a Vu window implements this.
 *        display() receives a CUDA device pointer (owned by the caller,
 *        i.e. gui.cu's interop layer) that the implementation fills in.
 */
class IRenderSource {
public:
    virtual ~IRenderSource() {}

    __HOST__ virtual int  width()  const = 0;
    __HOST__ virtual int  height() const = 0;
    __HOST__ virtual void display(TColor *d_dst)                      = 0;
    __HOST__ virtual void mouse(int button, int state, int x, int y)  {}
    __HOST__ virtual void keyboard(U8 k)                              {}
};

} // namespace t4::vu

#endif // __VU_RENDER_SOURCE_H
