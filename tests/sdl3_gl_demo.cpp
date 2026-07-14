//
// tensorForth tests/sdl3_gl_demo.cpp
//
// A more elaborate SDL3 smoke test than sdl3_demo.cpp: creates an OpenGL
// context (rather than an SDL_Renderer), and drives an animated triangle
// with keyboard + mouse input and delta-time-based movement. Meant as a
// closer approximation of what a real render loop in `vu` will look like
// once CUDA/GL interop is wired in — this file has none of that yet, it's
// just exercising SDL3's window/context/input/timing API surface.
//
// SDL3 API notes beyond what sdl3_demo.cpp covered:
//   - SDL_GL_SetAttribute() must be called *before* SDL_CreateWindow() with
//     SDL_WINDOW_OPENGL, to request context version/profile.
//   - SDL_GL_CreateContext() both creates and makes the context current.
//   - SDL_GL_DestroyContext() replaces SDL2's SDL_GL_DeleteContext().
//   - SDL_GetKeyboardState() returns `const bool*` in SDL3 (was `Uint8*`
//     in SDL2) — index it with SDL_SCANCODE_* for continuous key state,
//     as opposed to the discrete SDL_EVENT_KEY_DOWN/UP events used for
//     one-shot actions like Escape-to-quit.
//   - SDL_GetTicks() returns Uint64 in SDL3 (was Uint32 in SDL2).
//   - Mouse motion event coordinates (event.motion.x/y) are float, not int.
//
// This uses legacy immediate-mode GL (glBegin/glVertex) for simplicity, so
// it requests a compatibility-profile context. Real rendering code (in
// `vu`, eventually) will more likely want a core-profile context and
// shaders — swap the SDL_GL_CONTEXT_PROFILE_MASK attribute below when
// you get there.

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include <cstdio>
#include <cmath>

int main(int argc, char* argv[]) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    // Request a GL context BEFORE creating the window.
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* window = SDL_CreateWindow(
        "tensorForth — SDL3 GL demo (WASD/arrows to move, mouse to color)",
        900, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl = SDL_GL_CreateContext(window);
    if (!gl) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_GL_SetSwapInterval(1);   // vsync on

    int w, h;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    glViewport(0, 0, w, h);

    // Triangle position, driven by continuous keyboard state.
    float px = 0.0f, py = 0.0f;
    const float speed = 0.8f;   // units/sec in normalized device coords

    // Color, driven by mouse position.
    float r = 0.9f, g = 0.3f, b = 0.3f;

    Uint64 last_ticks = SDL_GetTicks();
    Uint64 frame_count = 0;
    Uint64 fps_timer   = last_ticks;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT:
                    running = false;
                    break;

                case SDL_EVENT_KEY_DOWN:
                    if (event.key.key == SDLK_ESCAPE) running = false;
                    break;

                case SDL_EVENT_WINDOW_RESIZED:
                    SDL_GetWindowSizeInPixels(window, &w, &h);
                    glViewport(0, 0, w, h);
                    break;

                case SDL_EVENT_MOUSE_MOTION:
                    // Map mouse position to a color, just to show motion
                    // events driving something visible.
                    r = event.motion.x / (float)w;
                    g = event.motion.y / (float)h;
                    b = 0.5f;
                    break;
            }
        }

        // Delta time, for frame-rate-independent movement.
        Uint64 now = SDL_GetTicks();
        float dt = (now - last_ticks) / 1000.0f;
        last_ticks = now;

        // Continuous keyboard state — WASD or arrow keys.
        const bool* keys = SDL_GetKeyboardState(nullptr);
        if (keys[SDL_SCANCODE_LEFT]  || keys[SDL_SCANCODE_A]) px -= speed * dt;
        if (keys[SDL_SCANCODE_RIGHT] || keys[SDL_SCANCODE_D]) px += speed * dt;
        if (keys[SDL_SCANCODE_UP]    || keys[SDL_SCANCODE_W]) py += speed * dt;
        if (keys[SDL_SCANCODE_DOWN]  || keys[SDL_SCANCODE_S]) py -= speed * dt;

        glClearColor(0.05f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBegin(GL_TRIANGLES);
            glColor3f(r, g, b);
            glVertex2f(px,        py + 0.2f);
            glVertex2f(px - 0.2f, py - 0.2f);
            glVertex2f(px + 0.2f, py - 0.2f);
        glEnd();

        SDL_GL_SwapWindow(window);

        // Update the window title with an FPS counter once a second.
        ++frame_count;
        if (now - fps_timer >= 1000) {
            char title[128];
            std::snprintf(title, sizeof(title),
                "tensorForth — SDL3 GL demo — %llu fps",
                (unsigned long long)frame_count);
            SDL_SetWindowTitle(window, title);
            frame_count = 0;
            fps_timer = now;
        }
    }

    SDL_GL_DestroyContext(gl);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
