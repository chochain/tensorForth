#
# tensorForth src/io/CMakeLists.txt
#
# Data-only fragment — same role as the old src/io/module.mk.

set(IO_SRCS
  aio.cpp
  aio_tensor.cpp
  aio_model.cpp
)

set(IO_DIRS)
set(IO_LIBS)

add_cuda_module(io ${CMAKE_CURRENT_SOURCE_DIR})
