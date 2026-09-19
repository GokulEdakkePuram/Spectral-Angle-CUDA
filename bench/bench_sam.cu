// bench_sam - kernel-level correctness and throughput for the SAM variants.
//
// Two jobs, in this order, because a fast wrong answer is worth nothing:
//
//   1. Check every variant against the double-precision CPU reference on the
//      same data, and report the worst angle error and any disagreement about
//      which target won.
//   2. Time each variant in isolation - no source, no upload, no detection -
//      and express the result as achieved DRAM bandwidth, which is the number
//      that says how close to the hardware's ceiling the kernel actually is.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hsi/frame_source.hpp"
#include "hsi/sam.hpp"
#include "hsi/sam_cuda.hpp"

namespace {

#define CUDA_OK(expr)                                                     \
  do {                                                                    \
    const cudaError_t err = (expr);                                       \
    if (err != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, \
                   cudaGetErrorString(err));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

struct Options {
  int height = 512;
  int width = 512;
  int bands = 128;
  int targets = 4;
  int iterations = 200;
  int warmup = 20;
  bool check_only = false;
  double tolerance = 2e-3;  ///< radians
  /// Pixels per thread for the Optimized kernel: 2, 4 or 8. 0 sweeps all three.
  int opt_pixels = 0;
};

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string token = argv[i];
    const std::size_t eq = token.find('=');
    const std::string key = (eq == std::string::npos) ? token : token.substr(0, eq);
    const std::string value = (eq == std::string::npos) ? "" : token.substr(eq + 1);
    if (key == "--height") options.height = std::stoi(value);
    else if (key == "--width") options.width = std::stoi(value);
    else if (key == "--bands") options.bands = std::stoi(value);
    else if (key == "--targets") options.targets = std::stoi(value);
    else if (key == "--iterations") options.iterations = std::stoi(value);
    else if (key == "--warmup") options.warmup = std::stoi(value);
    else if (key == "--tolerance") options.tolerance = std::stod(value);
    else if (key == "--opt-pixels") options.opt_pixels = std::stoi(value);
    else if (key == "--check-only") options.check_only = true;
    else if (key == "--help") {
      std::puts(
          "bench_sam [--height=N --width=N --bands=N --targets=N]\n"
          "          [--iterations=N --warmup=N --tolerance=RAD --check-only]\n"
          "          [--opt-pixels=2|4|8]   pixels per thread for `optimized`;\n"
          "                                 omit to time all three");
      std::exit(0);
    }
  }
  return options;
}

struct Buffers {
  float* d_bsq = nullptr;
  float* d_bip = nullptr;
  void* d_half = nullptr;
  float* d_targets = nullptr;
  float* d_norms = nullptr;
  float* d_angle = nullptr;
  std::int32_t* d_target_id = nullptr;
};

/// Worst-case angle error and label disagreement against the CPU reference.
struct Agreement {
  double max_angle_error = 0;
  std::size_t label_mismatches = 0;
  std::size_t significant_mismatches = 0;
};

/// A label mismatch only matters when the two candidates were meaningfully
/// apart. Where two targets sit at nearly the same angle from a pixel, which
/// one wins is a coin toss decided by the last bit of the accumulation, and
/// counting that as an error would just be measuring the rounding.
Agreement compare(const std::vector<float>& gpu_angle,
                  const std::vector<std::int32_t>& gpu_target,
                  const std::vector<float>& cpu_angle,
                  const std::vector<std::int32_t>& cpu_target,
                  const std::vector<float>& cpu_all, std::size_t pixels,
                  int targets, double tolerance) {
  Agreement result;
  for (std::size_t p = 0; p < pixels; ++p) {
    result.max_angle_error =
        std::max(result.max_angle_error,
                 static_cast<double>(std::fabs(gpu_angle[p] - cpu_angle[p])));
    if (gpu_target[p] == cpu_target[p]) continue;
    ++result.label_mismatches;
    const int g = gpu_target[p];
    if (g < 0 || g >= targets) {
      ++result.significant_mismatches;
      continue;
    }
    const double gap = std::fabs(cpu_all[static_cast<std::size_t>(g) * pixels + p] -
                                 cpu_angle[p]);
    if (gap > tolerance) ++result.significant_mismatches;
  }
  return result;
}

float time_variant(hsi::SamVariant variant, const Buffers& buffers,
                   hsi::CubeShape shape, int targets, int iterations, int warmup,
                   int opt_pixels = 0) {
  const void* cube = (variant == hsi::SamVariant::BipDirect) ? static_cast<const void*>(buffers.d_bip)
                     : (variant == hsi::SamVariant::Half)    ? buffers.d_half
                                                             : static_cast<const void*>(buffers.d_bsq);

  for (int i = 0; i < warmup; ++i) {
    CUDA_OK(hsi::launch_sam_best(variant, cube, shape, buffers.d_targets,
                                 buffers.d_norms, targets, buffers.d_angle,
                                 buffers.d_target_id, nullptr, opt_pixels));
  }
  CUDA_OK(cudaDeviceSynchronize());

  cudaEvent_t begin, end;
  CUDA_OK(cudaEventCreate(&begin));
  CUDA_OK(cudaEventCreate(&end));
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    CUDA_OK(hsi::launch_sam_best(variant, cube, shape, buffers.d_targets,
                                 buffers.d_norms, targets, buffers.d_angle,
                                 buffers.d_target_id, nullptr, opt_pixels));
  }
  CUDA_OK(cudaEventRecord(end));
  CUDA_OK(cudaEventSynchronize(end));

  float ms = 0;
  CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
  CUDA_OK(cudaEventDestroy(begin));
  CUDA_OK(cudaEventDestroy(end));
  return ms / static_cast<float>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse(argc, argv);

  hsi::CubeShape shape;
  shape.height = options.height;
  shape.width = options.width;
  shape.bands = options.bands;

  int device = 0;
  cudaDeviceProp props{};
  CUDA_OK(cudaGetDevice(&device));
  CUDA_OK(cudaGetDeviceProperties(&props, device));
  const double peak_gbs =
      2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6;
  std::printf("device %s  sm_%d%d  %d SMs  %.1f GB/s peak  %s\n", props.name,
              props.major, props.minor, props.multiProcessorCount, peak_gbs,
              props.integrated ? "integrated" : "discrete");
  std::printf("cube   %dx%dx%d  %d targets  %.1f MB/frame\n\n", shape.height,
              shape.width, shape.bands, options.targets,
              shape.bytes_f32() / 1.0e6);

  // Real-looking data: a mixed background with planted targets, so the branch
  // behaviour and the value range resemble what the detector actually sees.
  hsi::SyntheticOptions synth;
  synth.shape = shape;
  synth.num_targets = options.targets;
  auto source = hsi::make_synthetic_source(synth);

  const std::size_t pixels = shape.pixels();
  const std::size_t stride = hsi::bsq_plane_stride(shape);
  const std::size_t padded = hsi::bsq_device_elements(shape);

  std::vector<float> host_bsq(padded, 0.0f);
  hsi::FrameMeta meta;
  source->read_into(host_bsq.data(), stride, &meta);

  const hsi::SpectralLibrary& library = *source->library();
  const std::vector<float> flat = library.flatten();
  const std::vector<float> norms = library.norms();

  // The CPU reference wants a tightly packed cube; the device wants a padded
  // one. Strip the padding rather than teaching the reference about strides.
  std::vector<float> packed(shape.elements());
  for (int b = 0; b < shape.bands; ++b) {
    std::memcpy(packed.data() + static_cast<std::size_t>(b) * pixels,
                host_bsq.data() + static_cast<std::size_t>(b) * stride,
                pixels * sizeof(float));
  }
  std::vector<float> host_bip(shape.elements());
  hsi::convert_interleave(packed.data(), host_bip.data(), shape,
                          hsi::Interleave::Bsq, hsi::Interleave::Bip);

  Buffers buffers;
  CUDA_OK(cudaMalloc(&buffers.d_bsq, padded * sizeof(float)));
  CUDA_OK(cudaMalloc(&buffers.d_bip, shape.elements() * sizeof(float)));
  CUDA_OK(cudaMalloc(&buffers.d_half, padded * sizeof(short)));
  CUDA_OK(cudaMalloc(&buffers.d_targets, flat.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&buffers.d_norms, norms.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&buffers.d_angle, pixels * sizeof(float)));
  CUDA_OK(cudaMalloc(&buffers.d_target_id, pixels * sizeof(std::int32_t)));

  CUDA_OK(cudaMemcpy(buffers.d_bsq, host_bsq.data(), padded * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(buffers.d_bip, host_bip.data(),
                     shape.elements() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(buffers.d_targets, flat.data(), flat.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(buffers.d_norms, norms.data(), norms.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(hsi::launch_narrow_to_half(buffers.d_bsq, buffers.d_half, padded, nullptr));
  CUDA_OK(hsi::upload_targets(flat.data(), norms.data(), library.size(), shape.bands));
  CUDA_OK(cudaDeviceSynchronize());

  // ---- reference --------------------------------------------------------
  std::printf("computing the CPU reference (double precision)...\n");
  std::vector<float> cpu_angle(pixels);
  std::vector<std::int32_t> cpu_target(pixels);
  hsi::sam_best_cpu(packed.data(), shape, flat.data(), norms.data(),
                    library.size(), cpu_angle.data(), cpu_target.data());
  std::vector<float> cpu_all(pixels * static_cast<std::size_t>(library.size()));
  hsi::sam_cpu(packed.data(), shape, flat.data(), norms.data(), library.size(),
               cpu_all.data());

  // ---- correctness ------------------------------------------------------
  const hsi::SamVariant variants[] = {
      hsi::SamVariant::Baseline, hsi::SamVariant::Optimized,
      hsi::SamVariant::Half, hsi::SamVariant::BipDirect};

  std::printf("\n%-11s %14s %14s %s\n", "variant", "max angle err",
              "label mismatch", "verdict");
  std::printf("%s\n", std::string(64, '-').c_str());

  bool all_passed = true;
  for (hsi::SamVariant variant : variants) {
    const void* cube = (variant == hsi::SamVariant::BipDirect) ? static_cast<const void*>(buffers.d_bip)
                       : (variant == hsi::SamVariant::Half)    ? buffers.d_half
                                                               : static_cast<const void*>(buffers.d_bsq);
    CUDA_OK(cudaMemset(buffers.d_angle, 0, pixels * sizeof(float)));
    CUDA_OK(cudaMemset(buffers.d_target_id, 0xff, pixels * sizeof(std::int32_t)));
    CUDA_OK(hsi::launch_sam_best(variant, cube, shape, buffers.d_targets,
                                 buffers.d_norms, library.size(), buffers.d_angle,
                                 buffers.d_target_id, nullptr, options.opt_pixels));
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> gpu_angle(pixels);
    std::vector<std::int32_t> gpu_target(pixels);
    CUDA_OK(cudaMemcpy(gpu_angle.data(), buffers.d_angle, pixels * sizeof(float),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(gpu_target.data(), buffers.d_target_id,
                       pixels * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    // fp16 carries about three decimal digits, so it gets a looser bound -
    // stated here rather than hidden, because a detector's threshold has to be
    // set with the input precision's error in mind.
    const double tolerance = (variant == hsi::SamVariant::Half)
                                 ? std::max(options.tolerance, 5e-3)
                                 : options.tolerance;
    const Agreement agreement =
        compare(gpu_angle, gpu_target, cpu_angle, cpu_target, cpu_all, pixels,
                library.size(), tolerance);
    const bool passed = agreement.max_angle_error <= tolerance &&
                        agreement.significant_mismatches == 0;
    all_passed = all_passed && passed;

    std::printf("%-11s %14.2e %9zu (%zu) %s (tol %.1e)\n",
                hsi::to_string(variant), agreement.max_angle_error,
                agreement.label_mismatches, agreement.significant_mismatches,
                passed ? "PASS" : "FAIL", tolerance);
  }
  std::printf("\nlabel mismatch column is total (significant); a tie between two\n"
              "targets at the same angle is decided by rounding and is not an error.\n");

  if (options.check_only) {
    std::printf("\n%s\n", all_passed ? "all variants agree with the reference"
                                     : "SOME VARIANTS DISAGREE");
    return all_passed ? 0 : 1;
  }

  // ---- throughput -------------------------------------------------------
  std::printf("\n%-11s %10s %10s %10s %9s %s\n", "variant", "ms", "GB/s",
              "% of peak", "Gpix/s", "vs baseline");
  std::printf("%s\n", std::string(72, '-').c_str());

  float baseline_ms = 0;
  for (hsi::SamVariant variant : variants) {
    const float ms = time_variant(variant, buffers, shape, library.size(),
                                  options.iterations, options.warmup,
                                  options.opt_pixels);
    if (variant == hsi::SamVariant::Baseline) baseline_ms = ms;

    // Bytes the kernel must read from DRAM. The baseline re-reads the cube
    // once per target because its target loop is outermost; the vectorised
    // kernels read it once per pass of up to eight (four in fp16) targets.
    const double element_bytes = (variant == hsi::SamVariant::Half) ? 2.0 : 4.0;
    int passes = 1;
    if (variant == hsi::SamVariant::Baseline || variant == hsi::SamVariant::BipDirect) {
      passes = library.size();
    } else if (variant == hsi::SamVariant::Half) {
      passes = (library.size() + 3) / 4;
    } else {
      passes = (library.size() + 7) / 8;
    }
    const double bytes = static_cast<double>(shape.elements()) * element_bytes * passes;
    const double gbs = bytes / (ms * 1.0e-3) / 1.0e9;

    std::printf("%-11s %10.3f %10.1f %9.0f%% %9.2f %10.2fx\n",
                hsi::to_string(variant), ms, gbs,
                peak_gbs > 0 ? 100.0 * gbs / peak_gbs : 0.0,
                static_cast<double>(pixels) / (ms * 1.0e-3) / 1.0e9,
                baseline_ms > 0 ? baseline_ms / ms : 1.0);
  }

  // Pixels per thread for `optimized`, which changes how many constant-memory
  // reads ride on each vector load. At TT=8 that ratio is 8:1 at four pixels
  // and 4:1 at eight, and the sm_86 measurements say the ratio is what costs
  // the bandwidth - so this sweep is the experiment, not a tuning knob.
  if (options.opt_pixels == 0) {
    std::printf("\noptimized, pixels per thread (constant reads per vector load)\n");
    std::printf("%10s %10s %10s %10s %14s\n", "px/thread", "ms", "GB/s",
                "% of peak", "const/16B ld");
    std::printf("%s\n", std::string(60, '-').c_str());
    const int per_pass = std::min(library.size(), 8);
    for (int ppt : {2, 4, 8}) {
      const float ms = time_variant(hsi::SamVariant::Optimized, buffers, shape,
                                    library.size(), options.iterations,
                                    options.warmup, ppt);
      const int passes = (library.size() + 7) / 8;
      const double bytes = static_cast<double>(shape.elements()) * 4.0 * passes;
      const double gbs = bytes / (ms * 1.0e-3) / 1.0e9;
      // Constant-memory reads per 16 bytes loaded: TT reads ride on PPT/4 of
      // a float4, so two pixels per thread doubles the ratio and eight halves
      // it. This is the quantity the cliff tracks.
      std::printf("%10d %10.3f %10.1f %9.0f%% %14.1f\n", ppt, ms, gbs,
                  peak_gbs > 0 ? 100.0 * gbs / peak_gbs : 0.0,
                  per_pass / (ppt / 4.0));
    }
  }

  // The transpose that a BIP-delivering camera would need before the fast path.
  {
    cudaEvent_t begin, end;
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));
    for (int i = 0; i < options.warmup; ++i) {
      CUDA_OK(hsi::launch_transpose_bip_to_bsq(buffers.d_bip, buffers.d_bsq, shape, nullptr));
    }
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(begin));
    for (int i = 0; i < options.iterations; ++i) {
      CUDA_OK(hsi::launch_transpose_bip_to_bsq(buffers.d_bip, buffers.d_bsq, shape, nullptr));
    }
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
    ms /= static_cast<float>(options.iterations);
    const double bytes = 2.0 * static_cast<double>(shape.elements()) * 4.0;
    std::printf("\nbip->bsq transpose %.3f ms  %.1f GB/s (read + write)\n", ms,
                bytes / (ms * 1.0e-3) / 1.0e9);
    CUDA_OK(cudaEventDestroy(begin));
    CUDA_OK(cudaEventDestroy(end));
  }

  cudaFree(buffers.d_bsq);
  cudaFree(buffers.d_bip);
  cudaFree(buffers.d_half);
  cudaFree(buffers.d_targets);
  cudaFree(buffers.d_norms);
  cudaFree(buffers.d_angle);
  cudaFree(buffers.d_target_id);
  return all_passed ? 0 : 1;
}
