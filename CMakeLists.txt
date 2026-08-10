cmake_minimum_required(VERSION 3.20)
project(hyperspectral LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(HSI_BUILD_TESTS "Build unit tests" ON)
option(HSI_BUILD_CUDA  "Build the CUDA kernels and pipeline if a toolkit is found" ON)

# ---------------------------------------------------------------------------
# CUDA is optional so the host-side code (ENVI reader, CPU reference, frame
# sources) still builds and tests on a developer machine without a toolkit.
# ---------------------------------------------------------------------------
set(HSI_HAVE_CUDA OFF)
if(HSI_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HSI_HAVE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Default to a fat binary that covers Turing through Hopper, plus Orin
    # (sm_87), with PTX at the top end so unlisted future parts still run.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=87 for fast Jetson-only builds.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 90-virtual)
    endif()
  else()
    message(STATUS "No CUDA compiler found - building host-side targets only.")
  endif()
endif()

add_subdirectory(src)

if(HSI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(HSI_HAVE_CUDA)
  add_subdirectory(bench)
endif()

message(STATUS "hyperspectral: CUDA support = ${HSI_HAVE_CUDA}")
