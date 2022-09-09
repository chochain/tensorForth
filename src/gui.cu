/** -*- c++ -*- 
 * @File
 * @brief - tensorForth GUI - static, OpenGL in freeglut
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <map>
#include "gui.h"

#define REFRESH_DELAY     10              /** ms */
#define BUFFER_DATA(i)    ((char*)0 + i)

namespace T4GUI {
    
typedef std::map<int, Vu*> VuMap;
VuMap   vu_map;
GLuint  shader_id = 0;         ///< floating point shader

void _vu_set(int id, Vu *vu) {
    vu_map[id] = vu;
}

Vu *_vu_get() {
    int id = glutGetWindow();
    VuMap::iterator vu = vu_map.find(id);
    return (vu == vu_map.end()) ? NULL : vu->second;
}

// shader for displaying floating-point texture
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
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &xpos);
    if (xpos != -1) {
        const GLubyte *errmsg = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Shader error at: %d\n%s\n",  (int)xpos, errmsg);
    }
    printf("compiled\n");
}

void _cleanup() {
    Vu *vu = _vu_get();
    cudaGraphicsUnregisterResource(vu->pbo); GPU_CHK();
    
    if (vu_map.size()==0) {
        glDeleteProgramsARB(1, &shader_id);   /// remove shader
    }
}

void _gl_codepath(int w, int h) {
    // Common display code path
    glClear(GL_COLOR_BUFFER_BIT);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA,
        GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0);       /// texture coordinates:
    glVertex2f(-1, -1);       ///     (0,0) lower left
    glTexCoord2f(2, 0);       ///     (1,1) upper right
    glVertex2f(+3, -1);
    glTexCoord2f(0, 2);
    glVertex2f(-1, +3);
    glEnd();
    glFinish();
}

void _keyboard(unsigned char k, int /*x*/, int /*y*/) {
    switch (k) {
    case 27:     // ESC
    case 'q':
    case 'Q': glutDestroyWindow(glutGetWindow()); return;
    default: 
        Vu *vu = _vu_get();
        if (vu) vu->keyboard(k);
        break;
    }
}

void _display() {
    Vu  *vu = _vu_get();
    if (!vu) return;
    
    TColor *d_dst = NULL;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &vu->pbo, 0);   GPU_CHK();
    cudaGraphicsResourceGetMappedPointer(
        (void **)&d_dst, &num_bytes, vu->pbo);  GPU_CHK();

    vu->display(d_dst);
    
    cudaGraphicsUnmapResources(1, &vu->pbo, 0); GPU_CHK();
    _gl_codepath(vu->W, vu->H);
    
    glutSwapBuffers();
    glutReportErrors();
}

void _refresh(int) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, _refresh, 0);
    }
}

void _bind_texture(Vu *vu) {
    int    buf_sz = vu->W * vu->H * 4;
    GLuint gl_pbo, gl_tex;

    printf("\tTexture...");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 vu->W, vu->H, 0, GL_RGBA, GL_UNSIGNED_BYTE, vu->h_src);
    printf("created\n");

    printf("\tPBO...");
    glGenBuffers(1, &gl_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, buf_sz, vu->h_src, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it (once only).
    cudaGraphicsGLRegisterBuffer(
        &vu->pbo, gl_pbo, cudaGraphicsMapFlagsWriteDiscard);
    GPU_CHK();
    printf("created\n");
}

extern "C" int gui_init(int *argc, char **argv) {
    printf("\nGLUT...");
    glutInit(argc, argv);                /// * consumes X11 input parameters
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    printf("initialized\n");
    
    return 0;
}

extern "C" int gui_add(Vu *vu) {
    int z = 40 * vu_map.size();
    glutInitWindowPosition(600 + z - (vu->W / 2), 100 + z);
    glutInitWindowSize(vu->W, vu->H);
    ///
    /// create window for img
    ///
    printf("\tWindow...");
    int id = glutCreateWindow(T4_APP_NAME); /// * create named window (as current)
    _vu_set(id, vu);
    ///
    /// * set callbacks (for current window, i.e. id)
    ///
    glutDisplayFunc(_display);
    glutKeyboardFunc(_keyboard);
    glutTimerFunc(REFRESH_DELAY, _refresh, 0);
    glutCloseFunc(_cleanup);
    printf("created\n");

    _bind_texture(vu);
    _compile_shader();                      /// load float shader
    
    return 0;
}

extern "C" int gui_loop() {
    glutMainLoop();
    return 0;
}

} // namespace T4GUI
