/**
 * @file
 * @brief tensorForth - GUI helper funtions
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#ifndef T4_GUI_H
#define T4_GUI_H
#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <assert.h>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "vu.h"

namespace T4GUI {
#ifndef __GL_FUNC_EXTERN
#define __GL_FUNC_EXTERN
#define GLFN(f,intf) intf f = (intf)glXGetProcAddress((const GLubyte*)#f)
#else
#define GLFN(f,intf) extern intf f
#endif  // __GL_FUNC_EXTERN

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

    namespace Int {
        static std::vector<std::string> split(const std::string &str) {
            std::istringstream ss(str);
            std::istream_iterator<std::string> it(ss);
            return std::vector<std::string>(it, std::istream_iterator<std::string>());
        }
        /* Sort the vector passed by reference */
        template<typename T>
        static inline void sort(std::vector<T> &a) {
            std::sort(a.begin(), a.end());
        }
        /* Compare two vectors */
        template<typename T>
        static int equals(std::vector<T> a, std::vector<T> b) {
            if (a.size() != b.size()) return 0;
            sort(a);
            sort(b);
            return std::equal(a.begin(), a.end(), b.begin());
        }
        template<typename T>
        static std::vector<T> hit(std::vector<T> a, std::vector<T> b) {
            sort(a);
            sort(b);
            std::vector<T> rc;
            std::set_intersection(
                a.begin(), a.end(), b.begin(), b.end(),
                std::back_inserter<std::vector<std::string> >(rc));
            return rc;
        }
        static std::vector<std::string> gl_ext() {
            std::string ext((const char*)glGetString(GL_EXTENSIONS));
            return split(ext);
        }
    }
    static int glExtOK(const std::string &ext) {
        std::vector<std::string> all = Int::gl_ext();
        std::vector<std::string> req = Int::split(ext);
        std::vector<std::string> hit = Int::hit(all, req);
        return Int::equals(hit, req);
    }
    static int glVersionOK(int major, int minor) {
        std::string       ver((const char*)glGetString(GL_VERSION));
        std::stringstream ss(ver);
        int  m, n;
        char dot;
        ss >> m >> dot >> n;
        assert(dot == '.');
        return m > major || (m == major && n >= minor);
    }
    static inline const char* glErrString(GLenum err) {
        switch(err) {
        case GL_NO_ERROR:         return "GL_NO_ERROR";
        case GL_INVALID_ENUM:     return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:    return "GL_INVALID_VAL";
        case GL_INVALID_OPERATION:return "GL_INVALID_OP";
        case GL_OUT_OF_MEMORY:    return "GL_OUT_OF_MEMORY";
        case GL_STACK_UNDERFLOW:  return "GL_STACK_UNDERFLOW";
        case GL_STACK_OVERFLOW:   return "GL_STACK_OVERFLOW";
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FBO";
#endif
        default: return "UNKNOWN GL opcode";
        }
    }
    inline bool gl_check(const char *file, const int line) {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
            fprintf(stderr, "%s\n", glErrString(err));
            return false;
        }
        return true;
    }
#define GL_CHK()                         \
    if(!gl_check( __FILE__, __LINE__)) { \
        exit(EXIT_FAILURE);              \
    }
} /// namespace T4GUI

using namespace T4GUI;

#endif // T4_GUI_H
