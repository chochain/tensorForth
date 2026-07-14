#
# tensorForth src/ld/CMakeLists.txt
#
set(LD_SRCS
  loader.cpp
  mnist.cpp
  cifar10.cpp
)

set(LD_DIRS)
set(LD_LIBS)

add_cuda_module(ld ${CMAKE_CURRENT_SOURCE_DIR})
