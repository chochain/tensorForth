/** -*- c++ -*- 
 * @File
 * @brief - tensorForth GUI - static, implementated in OpenGL
 *
 * <pre>Copyright (C) 2022- GreenII, this file is distributed under BSD 3-Clause License.</pre>
 */
#include <map>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <helper_gl.h>                   /// from CUDA Samples
#include "imgvu.h"

namespace T4GUI {
    
typedef std::map<int, ImgVu*> VuMap;
VuMap vu_map;

// OpenGL PBO and texture "names"
struct cudaGraphicsResource *cuda_pbo;  ///< OpenGL-CUDA exchange
GLuint shader_id;                       ///< 

void _vu_set(int id, ImgVu *vu) {
    vu_map[id] = vu;
}

ImgVu *_vu_get(int id) {
    VuMap::iterator vu = vu_map.find(id);
    return (vu == vu_map.end()) ? NULL : vu->second;
}

// shader for displaying floating-point texture
void _compile_shader() {
    static const char *code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";
    
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
    cudaGraphicsUnregisterResource(cuda_pbo); GPU_CHK();
    glDeleteProgramsARB(1, &shader_id);   /// remove shader
}

void _gl_codepath(int h, int w) {
    // Common display code path
    glClear(GL_COLOR_BUFFER_BIT);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA,
        GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(2, 0);
    glVertex2f(+3, -1);
    glTexCoord2f(0, 2);
    glVertex2f(-1, +3);
    glEnd();
    glFinish();
}

void _display() {
    int   id  = glutGetWindow();
    ImgVu *vu = _vu_get(id);
    if (!vu) return;
    
    TColor *d_dst = NULL;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &cuda_pbo, 0);   GPU_CHK();
    cudaGraphicsResourceGetMappedPointer(
        (void **)&d_dst, &num_bytes, cuda_pbo);  GPU_CHK();

    vu->display(d_dst);
    
    cudaGraphicsUnmapResources(1, &cuda_pbo, 0); GPU_CHK();
    _gl_codepath(vu->height, vu->width);
    
    glutSwapBuffers();
    glutReportErrors();
}

void _refresh(int) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, _refresh, 0);
    }
}

void _keyboard(unsigned char k, int /*x*/, int /*y*/) {
    int id = glutGetWindow();
    
    switch (k) {
    case 27:     // ESC
    case 'q':
    case 'Q': glutDestroyWindow(id); return;
    default: 
        ImgVu *vu = _vu_get(id);
        if (vu) vu->keyboard(k);
        else {
            fprintf(stderr, "ImgVu[%d] not found\n", id);
            exit(-1);
        }
        break;
    }
}

void _init_opengl(uchar4 *h_src, int W, int H) {
    GLuint gl_pbo, gl_tex;
    int    buf_sz = W * H * 4;

    printf("\tTexture...");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_src);
    printf("created\n");

    printf("\tPBO...");
    glGenBuffers(1, &gl_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, buf_sz, h_src, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it (once only).
    cudaGraphicsGLRegisterBuffer(
        &cuda_pbo, gl_pbo, cudaGraphicsMapFlagsWriteDiscard);
    GPU_CHK();
    printf("created\n");
}

extern "C" int gui_init(int *argc, char **argv, ImgVu *vu, int x, int y) {
    static const char *ext =
        "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object";

    printf("\nGLUT...");
    glutInit(argc, argv);                /// * consumes X11 input parameters
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(vu->width, vu->height);
    glutInitWindowPosition(x - vu->width / 2, y - vu->height / 2);
    printf("initialized\n");
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
    
    if (!isGLVersionSupported(1, 5) ||
        !areGLExtensionsSupported(ext)) return -1;
    
    _init_opengl(vu->h_src, vu->width, vu->height);
    _compile_shader();                     /// load float shader
    
    return 0;
}

extern "C" int gui_loop() {
    glutMainLoop();
    return 0;
}


} // namespace T4GUI
