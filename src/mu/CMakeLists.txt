#
# tensorForth src/mu/CMakeLists.txt
#
set(MU_SRCS
  tlsf.cpp
  mpool.cpp
)

set(MU_CUSRCS
  tensor.cu
  mmu.cu
  dataset.cu
)

set(MU_DIRS)
set(MU_LIBS)

add_cuda_module(mu ${CMAKE_CURRENT_SOURCE_DIR})
