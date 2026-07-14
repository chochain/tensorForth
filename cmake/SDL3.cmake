#
# tensorForth cmake/SDL3.cmake
#
# Ubuntu 22.04's repos only ship SDL2 — SDL3 isn't packaged there yet.
# Strategy:
#   1. Look for a system/prefix install first (find_package, CONFIG mode —
#      SDL3 ships its own SDL3Config.cmake with proper imported targets).
#   2. If not found and SDL3_VENDORED is ON, fetch + build it from source
#      as part of this project's build tree.
#
# Either path leaves you with the SDL3::SDL3 imported target to link against.

option(SDL3_VENDORED
  "If SDL3 isn't found on the system, fetch and build it from source"
  ON
)

find_package(SDL3 QUIET CONFIG)

if(SDL3_FOUND)
  message(STATUS "SDL3: using system install (${SDL3_DIR})")
elseif(SDL3_VENDORED)
  message(STATUS "SDL3: not found on system, fetching source (release-3.2.x)")

  include(FetchContent)
  FetchContent_Declare(SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG        release-3.2.16   # pin explicitly; bump deliberately
    GIT_SHALLOW    TRUE
  )

  # Build SDL3 as a static lib by default to keep ten4_tests self-contained;
  # flip to shared if you'd rather ship libSDL3.so alongside the binary.
  set(SDL_SHARED OFF CACHE BOOL "" FORCE)
  set(SDL_STATIC ON  CACHE BOOL "" FORCE)
  set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)  # skip SDL's own test suite

  FetchContent_MakeAvailable(SDL3)

  # SDL's own CMake exports SDL3::SDL3 (aliased to SDL3-static when built
  # static), so downstream code can link SDL3::SDL3 unconditionally either way.
else()
  message(FATAL_ERROR
    "SDL3 not found and SDL3_VENDORED is OFF — install SDL3 or re-run with "
    "-DSDL3_VENDORED=ON to build it from source."
  )
endif()
