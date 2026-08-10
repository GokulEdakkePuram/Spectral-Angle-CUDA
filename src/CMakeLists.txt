# Host-only core: data types, ENVI I/O, frame sources, CPU reference SAM.
add_library(hsi_core
  types.cpp
  envi.cpp
  frame_source.cpp
  hot_loader.cpp
  sam_cpu.cpp
)
target_include_directories(hsi_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(hsi_core PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(hsi_core PUBLIC Threads::Threads)

if(NOT HSI_HAVE_CUDA)
  return()
endif()

# CUDA kernels and the streaming pipeline.
add_library(hsi_cuda
  sam.cu
  detect.cu
  pipeline.cu
)
target_link_libraries(hsi_cuda PUBLIC hsi_core)
set_target_properties(hsi_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(hsi_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

add_executable(hsi_detect main.cpp)
target_link_libraries(hsi_detect PRIVATE hsi_cuda)
