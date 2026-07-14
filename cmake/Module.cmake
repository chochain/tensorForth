#
# tensorForth cmake/Module.cmake
#
# Shared scaffolding for all sub-modules, equivalent to module_rules.mk's
# MODULE_RULE macro. Each src/<mod>/CMakeLists.txt sets <MOD>_SRCS,
# <MOD>_CUSRCS, <MOD>_DIRS, <MOD>_LIBS, then calls add_cuda_module().
#
# Compared to the Makefile version: no .cpp/.cu pattern rules to write by
# hand, and no -MMD -MP wiring — CMake + Ninja/Make track header deps
# automatically for both C++ and CUDA sources out of the box.

function(add_cuda_module NAME DIR)
  string(TOUPPER ${NAME} MOD)

  add_library(${NAME} OBJECT ${${MOD}_SRCS} ${${MOD}_CUSRCS})

  target_include_directories(${NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${${MOD}_DIRS}
  )

  target_link_libraries(${NAME} PUBLIC ${${MOD}_LIBS})

  set_target_properties(${NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  # Optional: mirrors the old `src-<mod>` phony convenience target.
  # `cmake --build build --target <mod>` already does this by default
  # for a library target, so no extra target is strictly needed.
endfunction()
