add_executable(bench_sam bench_sam.cu)
target_link_libraries(bench_sam PRIVATE hsi_cuda)

# NVML reports the current SM clock, which the benchmark prints beside every
# timing. Without it a throttled result is indistinguishable from a slow
# kernel. It is optional because Tegra's NVML support has historically been
# partial, and the benchmark degrades to printing 0.
if(TARGET CUDA::nvml)
  target_link_libraries(bench_sam PRIVATE CUDA::nvml)
  target_compile_definitions(bench_sam PRIVATE HSI_HAVE_NVML=1)
else()
  message(STATUS "NVML not found - bench_sam will not report SM clock")
endif()

# The benchmark doubles as the GPU correctness test: --check-only compares
# every variant against the CPU reference and exits non-zero on disagreement.
# It is registered against a small cube so it stays quick in CI.
if(HSI_BUILD_TESTS)
  add_test(NAME bench_sam_correctness
           COMMAND bench_sam --width=128 --height=96 --bands=24 --targets=4
                             --check-only)
endif()
