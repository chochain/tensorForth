#include <GL/glut.h>
#include <GL/glx.h>
#include <unistd.h>
#include <iostream>

#define PRINT(x) std::cout << x << std::endl;

void keyPressed1(unsigned char key, int x, int y) {
    PRINT("Key pressed in window 1")
}

void keyPressed2(unsigned char key, int x, int y) {
    PRINT("Key pressed in window 2")
}

int main(int argc, char ** argv) {
    // Draw green quad
    auto gl_path = [](float r, float g, float b) {
        PRINT("current glX drawable is " << glXGetCurrentDrawable());
        PRINT("current glX context is "  << glXGetCurrentContext());
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_QUADS);
        glColor3f(r, g, b);
        glVertex3f(-1.0f, 1.0f, 0.0f);
        glVertex3f( 1.0f, 1.0f, 0.0f);
        glVertex3f( 1.0f,-1.0f, 0.0f);
        glVertex3f(-1.0f,-1.0f, 0.0f);
        glEnd();
        glFinish();           // necessary
    };
    int sngBuf[] = {
        GLX_RGBA,             ///< 4 channels
        GLX_RED_SIZE,   1,
        GLX_GREEN_SIZE, 1,
        GLX_BLUE_SIZE,  1,
        GLX_DEPTH_SIZE, 12,   ///< depth buffer
        None
    };
    /// 1. Xlib creates windows
    /// 2. GLX hooks OpenGL <=> X
    /// 3. OpenGL draws
    Display * display = XOpenDisplay(0);
    XVisualInfo* vi   = glXChooseVisual(display, DefaultScreen(display), sngBuf);
    GLXContext glContext = glXCreateContext(display, vi, 0, GL_TRUE);

    // Create two windows
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(800, 800);

    glutCreateWindow("Window 1");
    glutKeyboardFunc(keyPressed1);
    GLXDrawable window1 = glXGetCurrentDrawable();
    PRINT("window 1 has ID: " << window1)

    glutInitWindowPosition(800, 0);
    glutCreateWindow("Window 2");
    glutKeyboardFunc(keyPressed2);
    GLXDrawable window2 = glXGetCurrentDrawable();
    PRINT("window 2 has ID: " << window2)

    usleep(100000); // have to wait here to make this work for some reason.
    // Maybe wait for windows to be created?

    // Render to window 1
    bool success = glXMakeCurrent(display, window1, glContext);
    if(!success) PRINT("failed to switch to window 1");
    gl_path(0.0f, 1.0f, 0.0f);

    // Render to window 2
    success = glXMakeCurrent(display, window2, glContext);
    if(!success) PRINT("failed to switch to window 2");
    gl_path(1.0f, 0.0f, 0.0f);

    glutMainLoop();
}
