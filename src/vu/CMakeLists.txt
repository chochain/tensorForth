#
# tensorForth src/vu/CMakeLists.txt
#
# PLACEHOLDER — copy the real VU_SRCS / VU_CUSRCS list over from your
# existing src/vu/module.mk. Structure and SDL3 wiring below should be
# correct as-is; only the file list needs filling in.

set(VU_SRCS
  # e.g. window.cpp
)

set(VU_CUSRCS
  # e.g. bmpvu.cu, glvu.cu
)

set(VU_DIRS)

# Assumption: `vu` owns SDL_CreateWindow / GL context creation / the event
# loop (i.e. what GLUT used to do), so it links SDL3 directly rather than
# only the test executable that consumes it. If that setup actually lives
# in tests/t_bmpvu.cu instead, move this SDL3 link there and drop it here.
set(VU_LIBS SDL3::SDL3)

add_cuda_module(vu ${CMAKE_CURRENT_SOURCE_DIR})
