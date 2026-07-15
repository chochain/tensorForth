#
# tensorForth root CMakeLists.txt
#
cmake_minimum_required(VERSION 3.22)   # 3.24+ for clean CUDA_ARCHITECTURES handling
project(ten4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architecture selection ─────────────────────────────────────────────
# Replaces the commented CUDA_ARCH/CUDA_CODE blocks in the old Makefile.
# Override at configure time instead of editing the build file, e.g.:
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=61         # GT 1030 (Pascal)
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=52         # Jetson Nano (Maxwell)
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=35         # older Kepler card
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES="75;61;52" # fat binary, all three
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)   # default: GTX 1660/1650 (Turing)
endif()

find_package(CUDAToolkit REQUIRED)

# ── Workaround: FindCUDAToolkit sometimes misses cudadevrt ─────────────────
# Known CMake bug (kitware issue #22279) — on some Ubuntu CUDA installs,
# CUDAToolkit_LIBRARY_DIR resolves in a way that doesn't include
# libcudadevrt.a even though the file is genuinely present. If the normal
# find_package() didn't create the target, look for it explicitly and
# build the imported target by hand so downstream `target_link_libraries(
# ... CUDA::cudadevrt)` calls keep working unmodified.
if(NOT TARGET CUDA::cudadevrt)
  find_library(CUDA_cudadevrt_LIBRARY
    NAMES cudadevrt
    PATHS
      ${CUDAToolkit_LIBRARY_DIR}
      ${CUDAToolkit_TARGET_DIR}/lib
      ${CUDAToolkit_TARGET_DIR}/lib64
    REQUIRED
  )
  add_library(CUDA::cudadevrt STATIC IMPORTED)
  set_target_properties(CUDA::cudadevrt PROPERTIES
    IMPORTED_LOCATION "${CUDA_cudadevrt_LIBRARY}"
  )
  message(STATUS "cudadevrt: FindCUDAToolkit missed it, added manually (${CUDA_cudadevrt_LIBRARY})")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ──────────────────────────────────────────────────────────
# Mirrors CC_FLAGS / NVCC_FLAGS from the old Makefile.
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-fomit-frame-pointer;-fno-stack-check;-fno-stack-protector;-march=native;-ffast-math;-funroll-loops>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;-Xptxas=-v;-lineinfo>"
)
set(CMAKE_CXX_FLAGS_RELEASE   "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE  "-O2")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)   # equivalent of --device-c
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Optional OpenGL/GLUT visualization (was commented GL_LIBS in Makefile) ──
option(WITH_GL "Enable OpenGL/GLUT visualization module" OFF)

# ── Shared module helper (equivalent of module_rules.mk) ───────────────────
include(cmake/Module.cmake)

# ── Keep vendored FetchContent deps outside build/ ──────────────────────────
# By default FetchContent downloads/builds under ${CMAKE_BINARY_DIR}/_deps,
# i.e. inside build/ — so `rm -rf build` or a whole-tree clean deletes SDL3's
# source and compiled static lib along with everything else, forcing a full
# re-clone + rebuild next time. Relocating this one directory outside build/
# means build/ can be nuked freely while SDL3 (and cutlass, once enabled)
# stay put. Add .deps/ to .gitignore alongside build/.
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/vendor" CACHE PATH
  "Where FetchContent downloads/builds vendored deps (kept outside build/)"
)

# ── SDL3 (visualization) ─────────────────────────────────────────────────────
# Finds a system SDL3 install, or builds it from source if not present
# (Ubuntu 22.04 doesn't package SDL3). Exposes the SDL3::SDL3 target.
include(cmake/SDL3.cmake)

# ── cutlass (GEMM/conv templates) ───────────────────────────────────────────
# Not in use yet — gated behind WITH_CUTLASS so a default configure neither
# fetches it nor requires CUTLASS_HOME to be set. Flip on with
# -DWITH_CUTLASS=ON once a module actually starts including cutlass headers.
# Uses CUTLASS_HOME if set (matching the old Makefile variable), otherwise
# fetches the header-only parts of the source tree. Exposes CUTLASS::CUTLASS.
option(WITH_CUTLASS "Enable cutlass GEMM/conv templates" OFF)
if(WITH_CUTLASS)
  include(cmake/Cutlass.cmake)
endif()

# ── Root-level sources (was src/module.mk) ──────────────────────────────────
add_subdirectory(src)

# ── Module registry ─────────────────────────────────────────────────────────
# To add a new module: drop src/<mod>/CMakeLists.txt, append the name here.
set(MODULES mu io vm ld nn tb vu)
foreach(mod ${MODULES})
  add_subdirectory(src/${mod})
endforeach()

# ── Top-level target ─────────────────────────────────────────────────────────
add_executable(${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE t4)
foreach(mod ${MODULES})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${mod})
endforeach()

# ── Tests / visualization executable (was tests/module.mk + `make tests`) ──
add_subdirectory(tests)

if(WITH_GL)
  find_package(OpenGL REQUIRED)
  find_package(X11 REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL OpenGL::GLU X11::X11 glut)
endif()

# `cmake -B build`                             create ~/build directory
# `cmake --build build --target ten4`       == old `make`
# `cmake --build build --target mu`         == old `make src-mu`
# `cmake --build build --target ten4_tests` == old `make tests`
# `cmake --build build --target clean`      == old `make clean` (whole-project
#                                              only; CMake has no free
#                                              per-module clean-mu equivalent)
# `ninja -C build -t clean mu`              == old `make clean-mu`
# `cmake --build build --target sdl3_gl_demo   build ~/tests/sdl3_gl_demo
