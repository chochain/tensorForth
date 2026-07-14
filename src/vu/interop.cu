/** -*- c++ -*-
 * @file
 * @brief GLInterop implementation - the only translation unit in this
 *        module that #includes both CUDA and GL headers.
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include "interop.h"
#include <cstdio>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace t4::vu {

int
GLInterop::register_pbo(unsigned int gl_pbo, bool write_discard) {
    if (_res) unregister_pbo();

    cudaGraphicsResource_t res = NULL;
    cudaError_t e = cudaGraphicsGLRegisterBuffer(
        &res, gl_pbo,
        write_discard ? cudaGraphicsMapFlagsWriteDiscard : cudaGraphicsMapFlagsNone);
    if (e != cudaSuccess) {
        fprintf(stderr, "GLInterop.register_pbo failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    _res = (void*)res;
    return 0;
}

void
GLInterop::unregister_pbo() {
    if (!_res) return;
    cudaGraphicsUnregisterResource((cudaGraphicsResource_t)_res);
    _res = NULL;
}

void*
GLInterop::map(size_t *bytes) {
    if (!_res) return NULL;
    cudaGraphicsResource_t res = (cudaGraphicsResource_t)_res;
    cudaGraphicsMapResources(1, &res, 0);

    void   *d_buf = NULL;
    size_t bsz    = 0;
    cudaGraphicsResourceGetMappedPointer(&d_buf, &bsz, res);
    if (bytes) *bytes = bsz;
    return d_buf;
}

void
GLInterop::unmap() {
    if (!_res) return;
    cudaGraphicsResource_t res = (cudaGraphicsResource_t)_res;
    cudaGraphicsUnmapResources(1, &res, 0);
}

} // namespace t4::vu
