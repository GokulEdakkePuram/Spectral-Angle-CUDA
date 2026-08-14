#include "hsi/sam.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hsi {
namespace {

/// Angle between one pixel and one target, given the target's precomputed
/// norm. Returns kMaxSamAngle when the pixel has no direction.
double angle_of(double dot, double pixel_sq, double target_norm) {
  if (pixel_sq <= 0.0 || target_norm <= 0.0) return kMaxSamAngle;
  double cosine = dot / (std::sqrt(pixel_sq) * target_norm);
  cosine = std::clamp(cosine, -1.0, 1.0);
  return std::acos(cosine);
}

}  // namespace

void sam_cpu(const float* cube_bsq, CubeShape shape, const float* targets,
             const float* target_norms, int num_targets, float* out_angle_rad) {
  const std::size_t pixels = shape.pixels();
  const int bands = shape.bands;

  for (std::size_t p = 0; p < pixels; ++p) {
    // Read the pixel's spectrum once, score it against every target. The
    // strided gather is the price of BSQ on a CPU; the GPU wants it the other
    // way round, which is exactly why the layout is pinned at the API edge.
    double pixel_sq = 0.0;
    for (int b = 0; b < bands; ++b) {
      const double v = cube_bsq[static_cast<std::size_t>(b) * pixels + p];
      pixel_sq += v * v;
    }
    for (int t = 0; t < num_targets; ++t) {
      double dot = 0.0;
      for (int b = 0; b < bands; ++b) {
        dot += static_cast<double>(cube_bsq[static_cast<std::size_t>(b) * pixels + p]) *
               targets[static_cast<std::size_t>(t) * bands + b];
      }
      out_angle_rad[static_cast<std::size_t>(t) * pixels + p] =
          static_cast<float>(angle_of(dot, pixel_sq, target_norms[t]));
    }
  }
}

void sam_best_cpu(const float* cube_bsq, CubeShape shape, const float* targets,
                  const float* target_norms, int num_targets,
                  float* out_angle_rad, std::int32_t* out_target) {
  const std::size_t pixels = shape.pixels();
  const int bands = shape.bands;

  for (std::size_t p = 0; p < pixels; ++p) {
    double pixel_sq = 0.0;
    for (int b = 0; b < bands; ++b) {
      const double v = cube_bsq[static_cast<std::size_t>(b) * pixels + p];
      pixel_sq += v * v;
    }
    double best = std::numeric_limits<double>::infinity();
    std::int32_t best_target = -1;
    for (int t = 0; t < num_targets; ++t) {
      double dot = 0.0;
      for (int b = 0; b < bands; ++b) {
        dot += static_cast<double>(cube_bsq[static_cast<std::size_t>(b) * pixels + p]) *
               targets[static_cast<std::size_t>(t) * bands + b];
      }
      const double a = angle_of(dot, pixel_sq, target_norms[t]);
      if (a < best) {
        best = a;
        best_target = t;
      }
    }
    out_angle_rad[p] = static_cast<float>(best);
    out_target[p] = best_target;
  }
}

namespace {

/// Mirrors is_local_best() in detect.cu, including the index tie-break.
bool local_best(const float* angle, int x, int y, CubeShape shape, int radius,
                float self) {
  const std::size_t self_index = static_cast<std::size_t>(y) * shape.width + x;
  for (int dy = -radius; dy <= radius; ++dy) {
    const int ny = y + dy;
    if (ny < 0 || ny >= shape.height) continue;
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) continue;
      const int nx = x + dx;
      if (nx < 0 || nx >= shape.width) continue;
      const std::size_t n = static_cast<std::size_t>(ny) * shape.width + nx;
      if (angle[n] < self) return false;
      if (angle[n] == self && n < self_index) return false;
    }
  }
  return true;
}

}  // namespace

std::vector<Detection> detect_cpu(const float* angle_rad,
                                  const std::int32_t* target, CubeShape shape,
                                  const DetectionParams& params) {
  std::vector<Detection> out;
  for (int y = 0; y < shape.height; ++y) {
    for (int x = 0; x < shape.width; ++x) {
      const std::size_t p = static_cast<std::size_t>(y) * shape.width + x;
      if (angle_rad[p] > params.threshold_rad) continue;
      if (params.nms_radius > 0 &&
          !local_best(angle_rad, x, y, shape, params.nms_radius, angle_rad[p])) {
        continue;
      }
      Detection d;
      d.x = x;
      d.y = y;
      d.target = target ? target[p] : 0;
      d.angle_rad = angle_rad[p];
      out.push_back(d);
    }
  }
  return out;
}

Spectrum mean_spectrum(const float* cube_bsq, CubeShape shape,
                       const std::uint8_t* mask, const char* name) {
  const std::size_t pixels = shape.pixels();
  std::size_t count = 0;
  std::vector<double> acc(static_cast<std::size_t>(shape.bands), 0.0);

  for (std::size_t p = 0; p < pixels; ++p) {
    if (!mask[p]) continue;
    ++count;
    for (int b = 0; b < shape.bands; ++b) {
      acc[static_cast<std::size_t>(b)] +=
          cube_bsq[static_cast<std::size_t>(b) * pixels + p];
    }
  }
  if (count == 0) {
    throw std::invalid_argument("mean_spectrum: mask selected no pixels");
  }

  Spectrum spectrum;
  spectrum.name = name ? name : "target";
  spectrum.values.resize(static_cast<std::size_t>(shape.bands));
  for (int b = 0; b < shape.bands; ++b) {
    spectrum.values[static_cast<std::size_t>(b)] =
        static_cast<float>(acc[static_cast<std::size_t>(b)] / static_cast<double>(count));
  }
  return spectrum;
}

}  // namespace hsi
