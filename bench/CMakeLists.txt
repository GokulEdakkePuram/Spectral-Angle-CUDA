add_executable(bench_sam bench_sam.cu)
target_link_libraries(bench_sam PRIVATE hsi_cuda)

# NVML is dlopen'd at runtime rather than linked, so only libdl is needed here.
# Linking CUDA::nvml would require nvml.h and the libnvidia-ml.so symlink, which
# live in a dev package that container images and JetPack images often omit even
# though the driver library itself is present.
target_link_libraries(bench_sam PRIVATE ${CMAKE_DL_LIBS})

# The benchmark doubles as the GPU correctness test: --check-only compares
# every variant against the CPU reference and exits non-zero on disagreement.
# It is registered against a small cube so it stays quick in CI.
if(HSI_BUILD_TESTS)
  add_test(NAME bench_sam_correctness
           COMMAND bench_sam --width=128 --height=96 --bands=24 --targets=4
                             --check-only)
endif()
