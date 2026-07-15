#
# tensorForth src/vu/CMakeLists.txt
#
set(VU_SRCS
  gui.cpp
)

set(VU_CUSRCS
  interop.cu
  vu.cu
  mnist_vu.cu
)

set(VU_DIRS)

# Assumption: `vu` owns SDL_CreateWindow / GL context creation / the event
# loop (i.e. what GLUT used to do), so it links SDL3 directly rather than
# only the test executable that consumes it. If that setup actually lives
# in tests/t_bmpvu.cu instead, move this SDL3 link there and drop it here.
set(VU_LIBS SDL3::SDL3)

add_cuda_module(vu ${CMAKE_CURRENT_SOURCE_DIR})
