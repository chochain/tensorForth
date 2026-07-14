#
# tensorForth tests/CMakeLists.txt
#
# Data fragment — same role as the old tests/module.mk — plus the
# visualization test executable that used to be the `make tests` target
# in the root Makefile.

set(TS_CUSRCS
  t_bmpvu.cu
  # t_imgvu.cu
  # t_mnist
  # t_ganvu
  # t_event
)

set(TS_DIRS)
set(TS_LIBS)

add_cuda_module(ts ${CMAKE_CURRENT_SOURCE_DIR})

# ── Visualization test executable ───────────────────────────────────────────
# Old rule (GLUT/X11 era):
#   tests: $(TS_CUOBJS:%:.o=)
#       $(NVCC) $(LINK_FLAGS) -lGL -lGLU -lglut -lX11 -lcudadevrt \
#           -o $(^:.o=) $^ $(VU_CUOBJS)
#
# SDL3 replaces GLUT + raw X11 for window creation, the GL context, and
# input — Xlib is no longer linked directly. OpenGL::GL is kept because
# rendering itself (glBegin/glDrawArrays/etc., or whatever `vu` does) is
# still raw GL calls made into the context SDL3 creates; SDL3 doesn't
# replace that part, only the windowing/event-loop/GLUT layer around it.
#
# Ubuntu 22.04 ships both the GLVND dispatch libraries (libOpenGL.so /
# libGLX.so) and, if legacy Mesa/NVIDIA packages are installed, the old
# monolithic libGL.so. CMake's FindOpenGL warns (policy CMP0072) if both
# are present and you don't say which to prefer. GLVND is the modern,
# vendor-neutral default on this distro (and required if you ever run
# with NVIDIA's proprietary driver alongside Mesa), so pin it explicitly
# rather than relying on link-order luck.
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# vu/gui.cu still calls GLUT directly (glutGetWindow, glutPostRedisplay,
# glutKeyboardFunc, etc.) — it hasn't actually been migrated to SDL3 for
# windowing/input yet, only assumed to be. Keep GLUT linked until that
# source-level migration happens; SDL3 being linked doesn't make existing
# GLUT calls resolve on its own.
find_package(GLUT REQUIRED)

add_executable(ten4_tests)
target_link_libraries(ten4_tests PRIVATE
  ts vu
  OpenGL::GL
  GLUT::GLUT
  SDL3::SDL3
  CUDA::cudadevrt
)
set_target_properties(ten4_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  CUDA_SEPARABLE_COMPILATION ON
)

# `cmake --build build --target ten4_tests`  == old `make tests`

# ── Standalone SDL3 smoke test ──────────────────────────────────────────────
# No dependency on ts/vu — just proves SDL3 is found/built and links
# correctly before you wire real rendering code into the vu module.
add_executable(sdl3_demo sdl3_demo.cpp)
target_link_libraries(sdl3_demo PRIVATE SDL3::SDL3)
set_target_properties(sdl3_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# `cmake --build build --target sdl3_demo` then run ./build/tests/sdl3_demo

# ── Standalone SDL3 + OpenGL demo ───────────────────────────────────────────
# More elaborate than sdl3_demo: GL context (not SDL_Renderer), keyboard +
# mouse input, delta-time animation, resize handling. Reuses the OpenGL
# find_package/GLVND preference already set up above for ten4_tests.
add_executable(sdl3_gl_demo sdl3_gl_demo.cpp)
target_link_libraries(sdl3_gl_demo PRIVATE SDL3::SDL3 OpenGL::GL)
set_target_properties(sdl3_gl_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# `cmake --build build --target sdl3_gl_demo` then run ./build/tests/sdl3_gl_demo
