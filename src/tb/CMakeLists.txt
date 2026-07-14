#
# tensorForth src/tb/CMakeLists.txt
#
set(TB_SRCS
  summary.cpp
)

set(TB_DIRS)
set(TB_LIBS)

add_cuda_module(tb ${CMAKE_CURRENT_SOURCE_DIR})
