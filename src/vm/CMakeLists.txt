#
# tensorForth src/vm/CMakeLists.txt
#
set(VM_SRCS
  vm.cpp
  eforth.cpp
  tenvm.cpp
  netvm.cpp
)

set(VM_DIRS)
set(VM_LIBS)

add_cuda_module(vm ${CMAKE_CURRENT_SOURCE_DIR})
