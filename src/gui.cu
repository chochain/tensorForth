/** -*- c++ -*- 
 * @File
 * @brief - tensorForth GUI - static, OpenGL in freeglut
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
GLuint  shader_id = 0;         ///< floating point shader

void _vu_set(int id, Vu *vu) {
    vu_map[id] = vu;
    printf("vu[%d] ", id);
}

Vu *_vu_get(int id) {
    VuMap::iterator it = vu_map.find(id);
    return (it == vu_map.end()) ? NULL : it->second;
}

Vu *_vu_now() {
    return _vu_get(glutGetWindow());
}
///
/// default texture shader for displaying floating-point
///
void _compile_shader() {
    static const char *code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";

    if (shader_id) return;    ///< already compiled
    
    printf("\tShader...");
    glGenProgramsARB(1, &shader_id);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader_id);
    glProgramStringARB(
        GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
        (GLsizei)strlen(code), (GLubyte*)code);
   
    GLint xpos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &xpos); /// CUDA GL extension
    if (xpos != -1) {
        const GLubyte *errmsg = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Shader error at: %d\n%s\n",  (int)xpos, errmsg);
    }
    printf("compiled\n");
}

void _close_and_switch_vu() {
    int id = glutGetWindow();
    Vu *vu = _vu_get(id);
    glutDestroyWindow(id);
    
    cudaGraphicsUnregisterResource(vu->pbo); GPU_CHK();
    vu_map.erase(id);                        /// * erase by key
    printf("\tvu[%d] released...", id);
    
    if (vu_map.size() > 0) {
        id = vu_map.rbegin()->first;
        glutSetWindow(id);                   /// * use another window
        printf("vu[%d] now active\n", id);
    }
    else printf("no avtive vu, shutting down...\n");
}

void _shutdown() {
    if (vu_map.size() > 0) return;
    
//    glDeleteProgramsARB(1, &shader_id);      /// remove shader
}

void _paint(int w, int h) {
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

void _mouse(int button, int state, int x, int y) {
    /// button: GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON
    /// state:  GLUT_UP=1, GLUT_DOWN=0
    /// x,y: mouse location in window relative coordinates
    switch (button) {
    case GLUT_LEFT_BUTTON:
    case GLUT_MIDDLE_BUTTON:
    case GLUT_RIGHT_BUTTON:
        Vu *vu = _vu_now();
        if (vu) vu->mouse(button, state, x, y);
        break;
    }
}

void _keyboard(unsigned char k, int /*x*/, int /*y*/) {
    switch (k) {
    case 27:     // ESC
    case 'q':
    case 'Q': _close_and_switch_vu(); break;
    default: 
        Vu *vu = _vu_now();
        if (vu) vu->keyboard(k);
        break;
    }
}

void _display() {
    Vu  *vu = _vu_now();
    if (!vu) return;
    
    TColor *d_dst = NULL;
    size_t bsz;

    cudaGraphicsMapResources(1, &vu->pbo, 0);   GPU_CHK();  /// lock
    cudaGraphicsResourceGetMappedPointer(
        (void**)&d_dst, &bsz, vu->pbo);         GPU_CHK();

    vu->display(d_dst);         /// update buffer content
    
    cudaGraphicsUnmapResources(1, &vu->pbo, 0); GPU_CHK();  /// unlock
    
    _paint(vu->X, vu->Y);
}

void _refresh(int) {
    if (!glutGetWindow()) return;
    
    glutPostRedisplay();       /// mark current window for refresh
    glutTimerFunc(T4_VU_REFRESH_DELAY, _refresh, 0);
}

void _bind_texture(Vu *vu) {
    GLuint gl_pbo, gl_tex;
    GLuint format = GL_RGBA8, depth = GL_RGBA;
    /*
    /// See OpenGL Core 3.2 internal format
    switch (vu->N) {
    case 1:  format = GL_R8;    depth = GL_RED;  break;
    case 2:  format = GL_RG8;   depth = GL_RG;   break;
    case 3:  format = GL_RGB8;  depth = GL_RGB;  break;
    default: format = GL_RGBA8; depth = GL_RGBA;
    }
    */
    printf("\tTexture");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, format,
                 vu->X, vu->Y, 0, depth, GL_UNSIGNED_BYTE, NULL);
    printf("[%d] created\n", gl_tex);

    printf("\tPBO");
    int bsz = vu->X * vu->Y * sizeof(uchar4);
    glGenBuffers(1, &gl_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, bsz, vu->h_tex, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it (once only).
    cudaGraphicsGLRegisterBuffer(
        &vu->pbo, gl_pbo, cudaGraphicsMapFlagsWriteDiscard);
    GPU_CHK();
    printf("[%d] created\n", gl_pbo);
}

extern "C" int gui_init(int *argc, char **argv) {
    printf("\nGLUT...");
    glutInit(argc, argv);                /// * consumes X11 input parameters
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    printf("initialized\n");
    
    return 0;
}

extern "C" int gui_add(Vu &vu) {
    int z = T4_VU_OFFSET * vu_map.size();
    glutInitWindowPosition(T4_VU_X_CENTER + z - (vu.X / 2), T4_VU_Y_CENTER + z);
    glutInitWindowSize(vu.X, vu.Y);
    ///
    /// create window for img
    ///
    printf("\tWindow...");
    int id = glutCreateWindow(T4_APP_NAME); /// * create named window (as current)
    _vu_set(id, &vu);
    ///
    /// * set callbacks (for current window, i.e. id)
    ///
    glutDisplayFunc(_display);
    glutKeyboardFunc(_keyboard);
    glutMouseFunc(_mouse);
    glutTimerFunc(T4_VU_REFRESH_DELAY, _refresh, 0);
    glutCloseFunc(_shutdown);
    printf("created\n");

    _bind_texture(&vu);
//    _compile_shader();                      /// load float shader
    
    return 0;
}

extern "C" int gui_loop() {
    glutMainLoop();
    return 0;
}

} // namespace T4GUI
