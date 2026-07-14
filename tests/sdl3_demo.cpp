//
// tensorForth tests/sdl3_demo.cpp
//
// Minimal standalone SDL3 example: creates a window, runs an event loop,
// and clears the screen to a solid color each frame. No dependency on
// the ts/vu modules — this is a self-contained sanity check that SDL3
// is found/built correctly and links cleanly under CMake.
//
// SDL3 API notes (differs from SDL2, in case you're used to that):
//   - SDL_Init() returns bool (true = success), not 0/-1 like SDL2.
//   - SDL_CreateWindow(title, w, h, flags) — no x/y position args anymore.
//   - SDL_CreateRenderer(window, driver_name) — driver name string or
//     nullptr for default, not an integer index.
//   - Event type constants are SDL_EVENT_* (e.g. SDL_EVENT_QUIT), not
//     the bare SDL_QUIT style from SDL2.
//   - SDL_GetError() is still how you retrieve the last error string.

#include <SDL3/SDL.h>
#include <cstdio>

int main(int argc, char* argv[]) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "tensorForth — SDL3 smoke test",
        800, 600,
        SDL_WINDOW_RESIZABLE
    );
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::printf("SDL3 window + renderer created. Close the window or press Esc to quit.\n");

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN &&
                       event.key.key == SDLK_ESCAPE) {
                running = false;
            }
        }

        // Clear to a dark teal each frame — swap this for your actual
        // GL/CUDA-interop render call once this smoke test passes.
        SDL_SetRenderDrawColor(renderer, 15, 40, 40, 255);
        SDL_RenderClear(renderer);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
