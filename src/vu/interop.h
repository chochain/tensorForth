/**
 * @file
 * @brief GLInterop - the single seam where CUDA and OpenGL meet.
 *
 * Every other file in this module is either pure-GL (gui.*) or pure-CUDA
 * (vu.*). Only interop.cu #includes both <cuda_gl_interop.h> and the GL
 * headers - this header itself stays free of both, so gui.h can include it
 * without pulling in the CUDA runtime.
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VU_INTEROP_H
#define __VU_INTEROP_H
#pragma once
#include <cstddef>

namespace t4::vu {

/**
 * @brief RAII wrapper around one CUDA<->GL pixel-buffer-object binding.
 *
 * The GL buffer id is passed in as a plain unsigned int (no GLuint), and
 * the CUDA resource handle is type-erased to void*, so this header never
 * needs <GL/gl.h> or <cuda_gl_interop.h>.
 */
class GLInterop {
public:
    GLInterop() {}
    ~GLInterop() { unregister_pbo(); }

    GLInterop(const GLInterop&)            = delete;   /// * owns a GPU resource, no copies
    GLInterop& operator=(const GLInterop&) = delete;

    /// register an existing GL_PIXEL_UNPACK_BUFFER_ARB object for CUDA write access
    int  register_pbo(unsigned int gl_pbo, bool write_discard=true);
    void unregister_pbo();

    /// map the buffer for CUDA access; returns a device pointer valid until unmap()
    void *map(size_t *bytes=NULL);
    void  unmap();

    bool  bound() const { return _res != NULL; }

private:
    void *_res = NULL;   ///< cudaGraphicsResource_t, type-erased
};

} // namespace t4::vu

#endif // __VU_INTEROP_H
