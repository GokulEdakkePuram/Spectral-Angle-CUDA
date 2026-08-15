#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hsi/frame_source.hpp"
#include "hsi/types.hpp"

namespace hsi {

/// A decoded greyscale PNG. Samples are normalised to [0, 1].
struct PngImage {
  int width = 0;
  int height = 0;
  std::vector<float> samples;  ///< row-major, width * height
};

/// Minimal PNG reader: 8 and 16-bit greyscale and RGB, non-interlaced.
///
/// Enough for snapshot-mosaic captures and nothing more. Colour images are
/// reduced to their first channel, which is what a mosaic sensor's output is
/// anyway when a tool has written it as RGB.
PngImage read_png_gray(const std::string& path);

/// Demosaic a snapshot-mosaic frame into a BSQ cube.
///
/// A snapshot mosaic sensor tiles a `period x period` grid of spectral filters
/// over the detector, so every `period x period` block of raw pixels is one
/// spatial pixel sampled at `period * period` wavelengths. Undoing that is a
/// pure de-interleave: raw (x, y) carries band (y % period) * period +
/// (x % period) of spatial pixel (x / period, y / period).
///
/// `band_order` optionally permutes mosaic positions into wavelength order;
/// leave it empty to keep raw mosaic order. Sensors do not lay their filters
/// out in increasing wavelength, and each dataset ships its own mapping, so
/// this has to be data rather than a constant.
HsiFrame demosaic(const PngImage& raw, int period, int bands,
                  const std::vector<int>& band_order = {});

/// Replays a HOT-style hyperspectral video: one directory of mosaic PNGs,
/// one PNG per frame, sorted by filename.
///
/// Written from the published description of the benchmark (XIMEA SSM 4x4 VIS,
/// 16 bands over 470-620 nm, 25 fps, 512x256 per band) because the data itself
/// is behind a request form at hsitracking.com. It has been tested against
/// synthetic mosaics of the same geometry, not against the real archive - so
/// check the first frame's band images look like a scene before trusting it.
struct HotOptions {
  std::string directory;
  int mosaic_period = 4;          ///< 4 for the 16-band VIS and RedNIR sets, 5 for 25-band NIR
  int bands = 16;                 ///< 16 (VIS), 15 (RedNIR, 4x4 with one dead position), 25 (NIR)
  std::vector<int> band_order;    ///< optional mosaic-position to wavelength permutation
  bool loop = true;
  int max_resident = 64;          ///< frames held in RAM; the rest stream from disk
};
std::unique_ptr<FrameSource> make_hot_source(const HotOptions& options);

}  // namespace hsi
