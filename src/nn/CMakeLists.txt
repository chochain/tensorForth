#
# tensorForth src/nn/CMakeLists.txt
#
set(NN_SRCS
  loss.cpp
  model.cpp
  )

set(NN_CUSRCS
  nmath.cu
  forward.cu
  backprop.cu
  debug.cu
  gradient.cu
)

set(NN_DIRS)
set(NN_LIBS)

add_cuda_module(nn ${CMAKE_CURRENT_SOURCE_DIR})
