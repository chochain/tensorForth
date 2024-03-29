/** -*- c++ -*- 
 * @file
 * @brief GUI helper - static, OpenGL in freeglut implementation
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <map>
#include "gui.h"

#define T4_VU_REFRESH_DELAY     100              /** ms     */
#define T4_VU_X_CENTER          600              /** pixels */
#define T4_VU_Y_CENTER          100              /** pixels */
#define T4_VU_OFFSET            40               /** pixels */

namespace T4GUI {
    
typedef std::map<int, Vu*> VuMap;
VuMap   vu_map;
GLuint  gl_pbo, gl_tex;        ///< GL pixel buffer object, texture
GLuint  gl_shader = 0;         ///< GL floating point shader

__HOST__ void _vu_set(int id, Vu *vu) { vu_map[id] = vu; }
__HOST__ Vu   *_vu_get(int id)        { return vu_map.find(id)->second; }
__HOST__ Vu   *_vu_curr()             { return _vu_get(glutGetWindow()); }
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
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &xpos); /// CUDA GL extension
    if (xpos != -1) {
        const GLubyte *errmsg = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Shader error at: %d\n%s\n",  (int)xpos, errmsg);
    }
    printf(": gl_shader[%d]", gl_shader);
}

__HOST__ void
_close_and_switch_vu() {
    int id  = glutGetWindow();
    Vu  *vu = _vu_get(id);
    glutDestroyWindow(id);

    if (vu->cu_pbo) {
        VUX(cudaGraphicsUnregisterResource(vu->cu_pbo));
    }
    vu_map.erase(id);                        /// * erase by key
    printf("\tvu.%d released...", id);
    
    if (vu_map.size() > 0) {
        id = vu_map.rbegin()->first;         /// * use another window
        glutSetWindow(id);
        printf("vu.%d now active\n", id);
    }
    else printf("no avtive vu\n");
}

__HOST__ void
_shutdown() {
    if (vu_map.size() > 0) return;
    if (gl_shader) glDeleteProgramsARB(1, &gl_shader);    /// release shader
    glDeleteBuffers(1, &gl_pbo);             /// * release GL video buffer
    
    printf("GLUT Done.\n");
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
        _vu_curr()->mouse(button, state, x, y);
        break;
    }
}

__HOST__ void
_keyboard(unsigned char k, int /*x*/, int /*y*/) {
    switch (k) {
    case 27:     // ESC
    case 'q':
    case 'Q': _close_and_switch_vu(); break;
    default:  _vu_curr()->keyboard(k); break;
    }
}

__HOST__ void
_display() {
    Vu *vu = _vu_curr();
    
    TColor *d_buf = NULL;
    size_t bsz;

    VUX(cudaGraphicsMapResources(1, &vu->cu_pbo, 0));   /// * lock CUDA vbo to GL buffer
    VUX(cudaGraphicsResourceGetMappedPointer(           /// * get device buffer pointer
            (void**)&d_buf, &bsz, vu->cu_pbo));
    printf("vu->cu_pbo=%p, d_buf=%p bsz=%ld\n", vu->cu_pbo, d_buf, bsz);
    
    if (d_buf) vu->display(d_buf);                      /// * update buffer content
    
    VUX(cudaGraphicsUnmapResources(1, &vu->cu_pbo, 0)); /// * unlock
    
    _paint(vu->X, vu->Y);                               /// * repaint GL
}

__HOST__ void
_refresh(int) {
    if (!glutGetWindow()) return;
    
    glutPostRedisplay();       /// mark current window for refresh
    glutTimerFunc(T4_VU_REFRESH_DELAY, _refresh, 0);
}

__HOST__ void
_bind_texture(Vu *vu) {
    const GLuint fmt = GL_RGBA8, depth = GL_RGBA;
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
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt,
                 vu->X, vu->Y, 0, depth, GL_UNSIGNED_BYTE, NULL);
    /*
    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    GLint  rst;
    while (rst!=GL_SIGNALED) {
        glSynciv(sync, GL_SYNC_STATUS, sizeof(rst), NULL, &rst);
    }
    */
    printf(", gl_tex[%d]", gl_tex);

    U64 bsz = vu->X * vu->Y * sizeof(uchar4);
    glGenBuffers(1, &gl_pbo);
    printf(", gl_pbo[%d] size=%ld", gl_pbo, bsz);
    ///
    /// stream h_tex to pbo
    ///
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, bsz, vu->h_tex, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it (once only).
    VUX(cudaGraphicsGLRegisterBuffer(
        &vu->cu_pbo, gl_pbo, cudaGraphicsMapFlagsWriteDiscard));
}

extern "C" int
gui_init(int *argc, char **argv) {
    printf("\nGLUT...");
    glutInit(argc, argv);                /// * consumes X11 input parameters
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    printf("initialized\n");
    
    return 0;
}

extern "C" int
gui_add(Vu *vu) {
    printf("gui_add Vu(%d,%d)", vu->X, vu->Y);
    
    int z = T4_VU_OFFSET * vu_map.size();
    glutInitWindowPosition(T4_VU_X_CENTER + z - (vu->X / 2), T4_VU_Y_CENTER + z);
    glutInitWindowSize(vu->X, vu->Y);
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
    /// * bind VU cuda texture to GL
    ///
    _compile_shader();                      /// load GL float shader
    _bind_texture(vu);                      /// * bind h_tex to GL buffer
    _vu_set(id, vu);                        /// * keep (id,vu&) pair in vu_map
    printf(" => vu.%d\n", id);
    
    return 0;
}

extern "C" int
gui_loop() {
    glutMainLoop();
    return 0;
}

} // namespace T4GUI
