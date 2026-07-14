/**
 * @file
 * @brief GUI helper - OpenGL funtions interface
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef __VU_GUI_H
#define __VU_GUI_H
#pragma once
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/freeglut.h>

#include "render_source.h"     /// * IRenderSource only - no CUDA, no Vu, no Corpus
#include "interop.h"           /// * GLInterop - CUDA-free by construction

namespace t4::vu {
///
/// GL extension function pointers.
/// Declared extern here, defined+resolved exactly once in gui.cu.
/// (Previously these were defined in this header behind a first-include
/// guard, which silently breaks - duplicate symbols at link time - the
/// moment gui.h is #included from a second .cu file.)
///
#define GLFN(f,intf) extern intf f
    GLFN(glBindBuffer,              PFNGLBINDBUFFERPROC);
    GLFN(glDeleteBuffers,           PFNGLDELETEBUFFERSPROC);
    GLFN(glBufferData,              PFNGLBUFFERDATAPROC);
    GLFN(glBufferSubData,           PFNGLBUFFERSUBDATAPROC);
    GLFN(glGenBuffers,              PFNGLGENBUFFERSPROC);
    GLFN(glCreateProgram,           PFNGLCREATEPROGRAMPROC);
    GLFN(glBindProgramARB,          PFNGLBINDPROGRAMARBPROC);
    GLFN(glGenProgramsARB,          PFNGLGENPROGRAMSARBPROC);
    GLFN(glDeleteProgramsARB,       PFNGLDELETEPROGRAMSARBPROC);
    GLFN(glDeleteProgram,           PFNGLDELETEPROGRAMPROC);
    GLFN(glGetProgramInfoLog,       PFNGLGETPROGRAMINFOLOGPROC);
    GLFN(glGetProgramiv,            PFNGLGETPROGRAMIVPROC);
    GLFN(glProgramParameteriEXT,    PFNGLPROGRAMPARAMETERIEXTPROC);
    GLFN(glProgramStringARB,        PFNGLPROGRAMSTRINGARBPROC);
    GLFN(glUnmapBuffer,             PFNGLUNMAPBUFFERPROC);
    GLFN(glMapBuffer,               PFNGLMAPBUFFERPROC);
    GLFN(glGetBufferParameteriv,    PFNGLGETBUFFERPARAMETERIVPROC);
    GLFN(glLinkProgram,             PFNGLLINKPROGRAMPROC);
    GLFN(glUseProgram,              PFNGLUSEPROGRAMPROC);
    GLFN(glAttachShader,            PFNGLATTACHSHADERPROC);
    GLFN(glCreateShader,            PFNGLCREATESHADERPROC);
    GLFN(glShaderSource,            PFNGLSHADERSOURCEPROC);
    GLFN(glCompileShader,           PFNGLCOMPILESHADERPROC);
    GLFN(glDeleteShader,            PFNGLDELETESHADERPROC);
    GLFN(glGetShaderInfoLog,        PFNGLGETSHADERINFOLOGPROC);
    GLFN(glGetShaderiv,             PFNGLGETSHADERIVPROC);
    GLFN(glUniform1i,               PFNGLUNIFORM1IPROC);
    GLFN(glUniform1f,               PFNGLUNIFORM1FPROC);
    GLFN(glUniform2f,               PFNGLUNIFORM2FPROC);
    GLFN(glUniform3f,               PFNGLUNIFORM3FPROC);
    GLFN(glUniform4f,               PFNGLUNIFORM4FPROC);
    GLFN(glUniform1fv,              PFNGLUNIFORM1FVPROC);
    GLFN(glUniform2fv,              PFNGLUNIFORM2FVPROC);
    GLFN(glUniform3fv,              PFNGLUNIFORM3FVPROC);
    GLFN(glUniform4fv,              PFNGLUNIFORM4FVPROC);
    GLFN(glUniformMatrix4fv,        PFNGLUNIFORMMATRIX4FVPROC);
    GLFN(glSecondaryColor3fv,       PFNGLSECONDARYCOLOR3FVPROC);
    GLFN(glGetUniformLocation,      PFNGLGETUNIFORMLOCATIONPROC);
    GLFN(glGenFramebuffersEXT,      PFNGLGENFRAMEBUFFERSEXTPROC);
    GLFN(glBindFramebufferEXT,      PFNGLBINDFRAMEBUFFEREXTPROC);
    GLFN(glDeleteFramebuffersEXT,   PFNGLDELETEFRAMEBUFFERSEXTPROC);
    GLFN(glCheckFramebufferStatusEXT, PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC);
    GLFN(glGetFramebufferAttachmentParameterivEXT, PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC);
    GLFN(glFramebufferTexture1DEXT, PFNGLFRAMEBUFFERTEXTURE1DEXTPROC);
    GLFN(glFramebufferTexture2DEXT, PFNGLFRAMEBUFFERTEXTURE2DEXTPROC);
    GLFN(glFramebufferTexture3DEXT, PFNGLFRAMEBUFFERTEXTURE3DEXTPROC);
    GLFN(glGenerateMipmapEXT,       PFNGLGENERATEMIPMAPEXTPROC);
    GLFN(glGenRenderbuffersEXT,     PFNGLGENRENDERBUFFERSEXTPROC);
    GLFN(glDeleteRenderbuffersEXT,  PFNGLDELETERENDERBUFFERSEXTPROC);
    GLFN(glBindRenderbufferEXT,     PFNGLBINDRENDERBUFFEREXTPROC);
    GLFN(glRenderbufferStorageEXT,  PFNGLRENDERBUFFERSTORAGEEXTPROC);
    GLFN(glFramebufferRenderbufferEXT, PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC);
    GLFN(glClampColorARB,           PFNGLCLAMPCOLORARBPROC);
    GLFN(glBindFragDataLocationEXT, PFNGLBINDFRAGDATALOCATIONEXTPROC);

#if !defined(GLX_EXTENSION_NAME) || !defined(GL_VERSION_1_3)
    GLFN(glActiveTexture,           PFNGLACTIVETEXTUREPROC);
    GLFN(glClientActiveTexture,     PFNGLACTIVETEXTUREPROC);
#endif
#undef GLFN
} /// namespace t4::vu

#endif // __VU_GUI_H
