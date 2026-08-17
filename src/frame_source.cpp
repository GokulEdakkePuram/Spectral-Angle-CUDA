#include "hsi/frame_source.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "hsi/envi.hpp"

namespace hsi {
namespace {

/// Cheap, reproducible noise. std::mt19937 per element would dominate the
/// generator's runtime and this only has to look like sensor noise.
inline std::uint32_t lcg(std::uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<std::uint32_t>(state >> 32);
}

inline float uniform(std::uint64_t& state) {
  return static_cast<float>(lcg(state)) * (1.0f / 4294967296.0f);
}

/// Sum of a few gaussians, which is roughly what a real reflectance curve
/// looks like: smooth, a handful of absorption and reflection features, and
/// nowhere near white noise.
std::vector<float> random_spectrum(int bands, std::uint64_t& state) {
  std::vector<float> out(static_cast<std::size_t>(bands), 0.0f);
  const int features = 3 + static_cast<int>(lcg(state) % 3);
  for (int f = 0; f < features; ++f) {
    const float centre = uniform(state) * static_cast<float>(bands);
    const float width = 0.05f * bands * (0.5f + uniform(state));
    const float height = 0.2f + 0.8f * uniform(state);
    for (int b = 0; b < bands; ++b) {
      const float d = (static_cast<float>(b) - centre) / width;
      out[static_cast<std::size_t>(b)] += height * std::exp(-0.5f * d * d);
    }
  }
  float peak = 0.0f;
  for (float v : out) peak = std::max(peak, v);
  if (peak <= 0.0f) peak = 1.0f;
  for (float& v : out) v = 0.05f + 0.85f * (v / peak);
  return out;
}

void parallel_for_bands(int bands, const std::function<void(int, int)>& body) {
  unsigned workers = std::thread::hardware_concurrency();
  if (workers == 0) workers = 1;
  workers = std::min<unsigned>(workers, static_cast<unsigned>(bands));
  if (workers <= 1) {
    body(0, bands);
    return;
  }
  std::vector<std::thread> pool;
  const int chunk = (bands + static_cast<int>(workers) - 1) / static_cast<int>(workers);
  for (unsigned w = 0; w < workers; ++w) {
    const int begin = static_cast<int>(w) * chunk;
    const int end = std::min(bands, begin + chunk);
    if (begin >= end) break;
    pool.emplace_back([&body, begin, end] { body(begin, end); });
  }
  for (std::thread& t : pool) t.join();
}

// ---------------------------------------------------------------------------

class EnviReplaySource final : public FrameSource {
 public:
  explicit EnviReplaySource(const EnviReplayOptions& options)
      : loop_(options.loop) {
    if (options.hdr_paths.empty()) {
      throw std::invalid_argument("EnviReplaySource: no headers given");
    }
    const int want = std::max(1, options.max_resident);
    for (const std::string& path : options.hdr_paths) {
      if (static_cast<int>(cubes_.size()) >= want) break;
      HsiFrame frame = read_envi(path, Interleave::Bsq);
      if (!shape_.valid()) {
        shape_ = frame.shape;
      } else if (frame.shape != shape_) {
        // A cube of a different size would silently break the whole pipeline's
        // preallocated buffers, so refuse it rather than reshaping.
        throw std::runtime_error("EnviReplaySource: " + path + " is " +
                                 std::to_string(frame.shape.height) + "x" +
                                 std::to_string(frame.shape.width) + "x" +
                                 std::to_string(frame.shape.bands) +
                                 ", expected the shape of the first cube");
      }
      names_.push_back(path);
      cubes_.push_back(std::move(frame.data));
    }
  }

  CubeShape shape() const override { return shape_; }
  Interleave interleave() const override { return Interleave::Bsq; }
  std::uint64_t length() const override {
    return loop_ ? 0 : static_cast<std::uint64_t>(cubes_.size());
  }

  bool read_into(float* dst, std::size_t plane_stride, FrameMeta* meta) override {
    if (cursor_ >= cubes_.size()) {
      if (!loop_) return false;
      cursor_ = 0;
    }
    const std::vector<float>& cube = cubes_[cursor_];
    const std::size_t pixels = shape_.pixels();
    if (plane_stride == pixels) {
      std::memcpy(dst, cube.data(), cube.size() * sizeof(float));
    } else {
      for (int b = 0; b < shape_.bands; ++b) {
        std::memcpy(dst + static_cast<std::size_t>(b) * plane_stride,
                    cube.data() + static_cast<std::size_t>(b) * pixels,
                    pixels * sizeof(float));
      }
    }
    if (meta) {
      meta->index = emitted_;
      meta->timestamp_s = static_cast<double>(emitted_) / 30.0;
    }
    ++cursor_;
    ++emitted_;
    return true;
  }

  void reset() override {
    cursor_ = 0;
    emitted_ = 0;
  }

  std::string describe() const override {
    return "envi-replay(" + std::to_string(cubes_.size()) + " cubes, " +
           std::to_string(shape_.height) + "x" + std::to_string(shape_.width) +
           "x" + std::to_string(shape_.bands) + ")";
  }

 private:
  bool loop_;
  CubeShape shape_;
  std::vector<std::string> names_;
  std::vector<std::vector<float>> cubes_;
  std::size_t cursor_ = 0;
  std::uint64_t emitted_ = 0;
};

// ---------------------------------------------------------------------------

class SyntheticSource final : public FrameSource {
 public:
  explicit SyntheticSource(const SyntheticOptions& options) : opt_(options) {
    if (!opt_.shape.valid()) throw std::invalid_argument("SyntheticSource: bad shape");

    std::uint64_t state = opt_.seed;
    const int bands = opt_.shape.bands;

    for (int i = 0; i < std::max(1, opt_.background_endmembers); ++i) {
      background_.push_back(random_spectrum(bands, state));
    }
    for (int i = 0; i < std::max(1, opt_.num_targets); ++i) {
      Spectrum s;
      s.name = "target" + std::to_string(i);
      s.values = random_spectrum(bands, state);
      library_.targets.push_back(std::move(s));
    }

    // The background assignment and per-pixel gain are fixed for the whole
    // sequence: a static scene with targets moving through it, which is the
    // case the detector has to hold up under.
    const std::size_t pixels = opt_.shape.pixels();
    material_.resize(pixels);
    gain_.resize(pixels);
    for (std::size_t p = 0; p < pixels; ++p) {
      material_[p] = static_cast<std::uint8_t>(lcg(state) % background_.size());
      gain_[p] = 1.0f + opt_.gain_jitter * (uniform(state) - 0.5f) * 2.0f;
    }

    for (int i = 0; i < std::max(1, opt_.num_targets); ++i) {
      Track track;
      track.x = uniform(state) * opt_.shape.width;
      track.y = uniform(state) * opt_.shape.height;
      track.vx = (uniform(state) - 0.5f) * 6.0f;
      track.vy = (uniform(state) - 0.5f) * 6.0f;
      tracks_.push_back(track);
    }

    mask_.assign(pixels, 0);
    noise_state_ = opt_.seed ^ 0x9e3779b97f4a7c15ULL;
  }

  CubeShape shape() const override { return opt_.shape; }
  Interleave interleave() const override { return Interleave::Bsq; }
  std::uint64_t length() const override { return opt_.length; }
  const std::uint8_t* truth_mask() const override { return mask_.data(); }
  const SpectralLibrary* library() const override { return &library_; }

  bool read_into(float* dst, std::size_t plane_stride, FrameMeta* meta) override {
    if (opt_.length != 0 && emitted_ >= opt_.length) return false;

    const CubeShape shape = opt_.shape;
    const std::size_t pixels = shape.pixels();

    // Background first, written band plane by band plane so every store is
    // sequential. Each worker keeps its own noise stream, seeded from the band
    // index, so the output does not depend on how the work got split.
    parallel_for_bands(shape.bands, [&](int b0, int b1) {
      for (int b = b0; b < b1; ++b) {
        std::uint64_t state = noise_state_ + static_cast<std::uint64_t>(b) * 1000003ULL +
                              emitted_ * 7919ULL;
        float* plane = dst + static_cast<std::size_t>(b) * plane_stride;
        for (std::size_t p = 0; p < pixels; ++p) {
          const float base = background_[material_[p]][static_cast<std::size_t>(b)];
          const float noise = (uniform(state) - 0.5f) * 2.0f * opt_.noise_sigma;
          plane[p] = base * gain_[p] + noise;
        }
      }
    });

    std::fill(mask_.begin(), mask_.end(), std::uint8_t{0});

    // Then stamp the targets over it. Bouncing keeps them in frame without
    // teleporting, so a tracker downstream sees continuous motion.
    const int radius = std::max(1, opt_.target_radius);
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
      Track& track = tracks_[t];
      track.x += track.vx;
      track.y += track.vy;
      if (track.x < radius || track.x > shape.width - radius - 1) {
        track.vx = -track.vx;
        track.x = std::clamp(track.x, static_cast<float>(radius),
                             static_cast<float>(shape.width - radius - 1));
      }
      if (track.y < radius || track.y > shape.height - radius - 1) {
        track.vy = -track.vy;
        track.y = std::clamp(track.y, static_cast<float>(radius),
                             static_cast<float>(shape.height - radius - 1));
      }

      const std::vector<float>& spectrum = library_.targets[t].values;
      const int cx = static_cast<int>(track.x);
      const int cy = static_cast<int>(track.y);
      std::uint64_t state = noise_state_ + t * 31ULL + emitted_ * 104729ULL;

      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          if (dx * dx + dy * dy > radius * radius) continue;
          const int x = cx + dx;
          const int y = cy + dy;
          if (x < 0 || x >= shape.width || y < 0 || y >= shape.height) continue;
          const std::size_t p = static_cast<std::size_t>(y) * shape.width + x;
          mask_[p] = static_cast<std::uint8_t>(t + 1);
          const float g = gain_[p];
          for (int b = 0; b < shape.bands; ++b) {
            const float noise = (uniform(state) - 0.5f) * 2.0f * opt_.noise_sigma;
            dst[static_cast<std::size_t>(b) * plane_stride + p] =
                spectrum[static_cast<std::size_t>(b)] * g + noise;
          }
        }
      }
    }

    if (meta) {
      meta->index = emitted_;
      meta->timestamp_s = static_cast<double>(emitted_) / 30.0;
    }
    ++emitted_;
    return true;
  }

  void reset() override { emitted_ = 0; }

  std::string describe() const override {
    return "synthetic(" + std::to_string(opt_.shape.height) + "x" +
           std::to_string(opt_.shape.width) + "x" +
           std::to_string(opt_.shape.bands) + ", " +
           std::to_string(tracks_.size()) + " targets)";
  }

 private:
  struct Track {
    float x = 0, y = 0, vx = 0, vy = 0;
  };

  SyntheticOptions opt_;
  std::vector<std::vector<float>> background_;
  SpectralLibrary library_;
  std::vector<std::uint8_t> material_;
  std::vector<float> gain_;
  std::vector<Track> tracks_;
  std::vector<std::uint8_t> mask_;
  std::uint64_t noise_state_ = 0;
  std::uint64_t emitted_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> make_envi_replay_source(const EnviReplayOptions& options) {
  return std::make_unique<EnviReplaySource>(options);
}

std::unique_ptr<FrameSource> make_synthetic_source(const SyntheticOptions& options) {
  return std::make_unique<SyntheticSource>(options);
}

}  // namespace hsi
