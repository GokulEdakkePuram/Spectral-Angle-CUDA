#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hsi/types.hpp"

namespace hsi {

struct FrameMeta {
  std::uint64_t index = 0;
  double timestamp_s = 0.0;
};

/// A stream of hyperspectral frames.
///
/// Sources write into caller-provided memory rather than returning a frame
/// they own. The pipeline allocates those buffers as pinned - and on the Orin,
/// as pinned *and* device-mapped - so the source's write is the only time the
/// cube is touched on the host. A source that owned its own buffer would force
/// a second 100+ MB memcpy per frame into staging, which at video rate is
/// enough on its own to miss the frame budget.
class FrameSource {
 public:
  virtual ~FrameSource() = default;

  virtual CubeShape shape() const = 0;
  virtual Interleave interleave() const = 0;

  /// Number of distinct frames before the stream repeats or ends.
  /// 0 means unbounded.
  virtual std::uint64_t length() const = 0;

  /// Write the next frame into `dst` as BSQ, leaving `plane_stride` elements
  /// between the start of one band plane and the next. Pass shape().pixels()
  /// for a tightly packed cube.
  ///
  /// The stride is here for the Orin's zero-copy path: the GPU reads the host
  /// buffer in place, so the buffer has to already carry the padded plane
  /// stride the kernels need for aligned vector loads. Without it, zero-copy
  /// would need a repacking pass and stop being zero-copy.
  ///
  /// Returns false once the stream is exhausted.
  virtual bool read_into(float* dst, std::size_t plane_stride, FrameMeta* meta) = 0;

  virtual void reset() = 0;
  virtual std::string describe() const = 0;

  /// Ground-truth target mask for the frame most recently written, one byte
  /// per pixel in raster order, or null if the source has no ground truth.
  virtual const std::uint8_t* truth_mask() const { return nullptr; }

  /// Spectra the source knows are present, for sources that plant their own
  /// targets. Null when the signatures have to come from elsewhere.
  virtual const SpectralLibrary* library() const { return nullptr; }
};

/// Replays ENVI cubes from disk, holding a bounded number of them resident.
///
/// HyperBlood cubes are ~160 MB each, so keeping all fourteen in RAM is
/// usually not what you want; `max_resident` caps how many are loaded and the
/// source cycles through those.
struct EnviReplayOptions {
  std::vector<std::string> hdr_paths;
  int max_resident = 4;
  bool loop = true;
};
std::unique_ptr<FrameSource> make_envi_replay_source(const EnviReplayOptions& options);

/// Generates hyperspectral video with known targets moving through a mixed
/// background.
///
/// This exists because the real datasets that ship with ground truth are
/// single scenes, not video, and because throughput has to be measured at
/// resolutions and band counts no public dataset happens to provide.
struct SyntheticOptions {
  CubeShape shape{512, 512, 128};
  int background_endmembers = 6;
  int num_targets = 3;
  int target_radius = 6;
  float gain_jitter = 0.4f;   ///< multiplicative, the illumination SAM ignores
  float noise_sigma = 0.01f;  ///< additive, the part it cannot
  std::uint64_t seed = 20260919;
  std::uint64_t length = 0;   ///< 0 == unbounded
};
std::unique_ptr<FrameSource> make_synthetic_source(const SyntheticOptions& options);

}  // namespace hsi
