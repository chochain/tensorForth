/** -*- c++ -*- 
 * @file
 * @brief GUI helper - static, OpenGL in freeglut implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <cstdio>
#include <cstring>
#include <map>
#include "gui.h"

#define T4_VU_REFRESH_DELAY     100              /** ms     */
#define T4_VU_X_CENTER          600              /** pixels */
#define T4_VU_Y_CENTER          100              /** pixels */
#define T4_VU_OFFSET            40               /** pixels */

namespace t4::vu {
///
/// GL extension function pointers - defined + resolved exactly once, here.
/// (gui.h only declares them `extern`.)
///
#define GLFN(f,intf) intf f = (intf)glXGetProcAddress((const GLubyte*)#f)
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

///
/// Per-window GL state. Previously gl_pbo/gl_tex were single globals shared
/// (i.e. clobbered) across every open Vu window; now each window owns its
/// own texture, pbo, and CUDA-GL interop binding.
///
struct _Win {
    IRenderSource *src;      ///< the thing this window renders
    GLInterop     interop;   ///< this window's CUDA<->GL pbo binding
    GLuint        pbo = 0;
    GLuint        tex = 0;
};

typedef std::map<int, _Win*> VuMap;
VuMap   vu_map;
GLuint  gl_shader = 0;         ///< GL floating point shader (shared, stateless)

__HOST__ _Win *_vu_get(int id)  { return vu_map.find(id)->second; }
__HOST__ _Win *_vu_curr()       { return _vu_get(glutGetWindow()); }
///
/// default texture shader for displaying floating-point
///
__HOST__ void
_compile_shader() {
    static const char *code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";

    if (gl_shader) return;    ///< already compiled

    glGenProgramsARB(1, &gl_shader);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_shader);
    glProgramStringARB(
        GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
        (GLsizei)strlen(code), (GLubyte*)code);

    GLint xpos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &xpos);
    if (xpos != -1) {
        const GLubyte *errmsg = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Shader error at: %d\n%s\n",  (int)xpos, errmsg);
    }
    printf(": gl_shader[%d]", gl_shader);
}

__HOST__ void
_release_window(_Win *w) {
    w->interop.unregister_pbo();             /// * safe even if never registered
    if (w->pbo) glDeleteBuffers(1, &w->pbo);
    if (w->tex) glDeleteTextures(1, &w->tex);
    delete w;
}

__HOST__ void
_shutdown() {
    int  id = glutGetWindow();
    _Win *w = _vu_get(id);
    _release_window(w);
    vu_map.erase(id);
    printf("\tvu.%d released...", id);

    if (vu_map.size() > 0) {
        int nid = vu_map.rbegin()->first;    /// * switch to another open window
        glutSetWindow(nid);
        printf("vu.%d now active\n", nid);
    }
    else {
        if (gl_shader) { glDeleteProgramsARB(1, &gl_shader); gl_shader = 0; }
        printf("no active vu, GLUT Done.\n");
    }
}

__HOST__ void
_paint(int w, int h) {
    // Common display code path
    glClear(GL_COLOR_BUFFER_BIT);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, w, h,
        GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0);       /// texture coordinates:
    glVertex2f(-1, -1);       ///     (0,0) lower left
    glTexCoord2f(2, 0);       ///     (1,1) upper right
    glVertex2f(+3, -1);
    glTexCoord2f(0, 2);
    glVertex2f(-1, +3);
    glEnd();
    glFinish();

    glutSwapBuffers();
    glutReportErrors();
}

__HOST__ void
_mouse(int button, int state, int x, int y) {
    /// button: GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON
    /// state:  GLUT_UP=1, GLUT_DOWN=0
    /// x,y: mouse location in window relative coordinates
    switch (button) {
    case GLUT_LEFT_BUTTON:
    case GLUT_MIDDLE_BUTTON:
    case GLUT_RIGHT_BUTTON:
        _vu_curr()->src->mouse(button, state, x, y);
        break;
    }
}

__HOST__ void
_keyboard(unsigned char k, int /*x*/, int /*y*/) {
    switch (k) {
    case 27:     // ESC
    case 'q':
    case 'Q': glutDestroyWindow(glutGetWindow()); break;  /// * triggers _shutdown
    default:  _vu_curr()->src->keyboard(k); break;
    }
}

__HOST__ void
_display() {
    _Win *w = _vu_curr();

    size_t bsz;
    TColor *d_buf = (TColor*)w->interop.map(&bsz);   /// * lock CUDA vbo to GL buffer

    if (d_buf) w->src->display(d_buf);               /// * update buffer content

    w->interop.unmap();                               /// * unlock

    _paint(w->src->width(), w->src->height());        /// * repaint GL
}

__HOST__ void
_refresh(int) {
    if (!glutGetWindow()) return;

    glutPostRedisplay();       /// mark current window for refresh
    glutTimerFunc(T4_VU_REFRESH_DELAY, _refresh, 0);
}

__HOST__ void
_bind_texture(_Win *w) {
    const GLuint fmt = GL_RGBA8, depth = GL_RGBA;
    const int W = w->src->width(), H = w->src->height();
    /*
    /// See OpenGL Core 3.2 internal format
    switch (vu->N) {
    case 1:  fmt = GL_R8;    depth = GL_RED;  break;
    case 2:  fmt = GL_RG8;   depth = GL_RG;   break;
    case 3:  fmt = GL_RGB8;  depth = GL_RGB;  break;
    default: fmt = GL_RGBA8; depth = GL_RGBA;
    }
    */
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &w->tex);
    glBindTexture(GL_TEXTURE_2D, w->tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, W, H, 0, depth, GL_UNSIGNED_BYTE, NULL);
    printf(", gl_tex[%d]", w->tex);

    size_t bsz = (size_t)W * H * sizeof(TColor);
    glGenBuffers(1, &w->pbo);
    printf(", gl_pbo[%d] size=%zu", w->pbo, bsz);
    ///
    /// Allocate the pbo. Initial contents are irrelevant - it's about to be
    /// registered with write-discard, and CUDA fills it every frame - so
    /// gui.cu no longer needs to reach into Vu's host texture to seed it.
    ///
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, w->pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, bsz, NULL, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it (once only).
    w->interop.register_pbo(w->pbo);
}

extern "C" int
gui_add(IRenderSource *src) {
    printf("gui_add Vu(%d,%d)", src->width(), src->height());

    _Win *w = new _Win();
    w->src  = src;

    int z = T4_VU_OFFSET * (int)vu_map.size();
    glutInitWindowPosition(T4_VU_X_CENTER + z - (src->width() / 2), T4_VU_Y_CENTER + z);
    glutInitWindowSize(src->width(), src->height());
    ///
    /// create GL window
    ///
    int id = glutCreateWindow(T4_APP_NAME); /// * create named window (as current)
    ///
    /// * set callbacks (for current window, i.e. id)
    ///
    glutDisplayFunc(_display);
    glutKeyboardFunc(_keyboard);
    glutMouseFunc(_mouse);
    glutTimerFunc(T4_VU_REFRESH_DELAY, _refresh, 0);
    glutCloseFunc(_shutdown);
    ///
    /// * bind this window's texture/pbo and CUDA-GL interop
    ///
    _compile_shader();                      /// load GL float shader
    _bind_texture(w);                       /// * bind texture/pbo to this window
    vu_map[id] = w;                         /// * keep (id, window) pair in vu_map
    printf(" => vu.%d\n", id);

    return 0;
}

} // namespace t4::vu

extern "C" int
gui_init(int *argc, char **argv) {
    printf("\nGLUT...");
    glutInit(argc, argv);                /// * consumes X11 input parameters
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    printf("initialized\n");

    return 0;
}

extern "C" int
gui_loop() {
    glutMainLoop();
    return 0;
}
