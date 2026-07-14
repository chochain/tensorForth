#
# tensorForth src/CMakeLists.txt
#
# Root-level sources — same role as the old src/module.mk. These files
# (sys.cpp, debug.cpp, util.cu, t4math.cu, ten4.cu) link directly into the
# main ten4 binary, same as every other module.

set(T4_SRCS
  sys.cpp
  debug.cpp
)

set(T4_CUSRCS
  util.cu
  t4math.cu
  ten4.cu
)

set(T4_DIRS)
set(T4_LIBS)

add_cuda_module(t4 ${CMAKE_CURRENT_SOURCE_DIR})
