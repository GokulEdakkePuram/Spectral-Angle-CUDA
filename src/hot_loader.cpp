#include "hsi/hot_loader.hpp"

#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace hsi {
namespace {

std::uint32_t read_be32(const unsigned char* p) {
  return (static_cast<std::uint32_t>(p[0]) << 24) |
         (static_cast<std::uint32_t>(p[1]) << 16) |
         (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

int paeth(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = std::abs(p - a);
  const int pb = std::abs(p - b);
  const int pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/// Undo the per-scanline filter PNG applies before compression. Each line
/// carries its filter type as a leading byte and predicts from the pixel to
/// the left, the line above, or both.
void unfilter(std::vector<unsigned char>& raw, int height, std::size_t stride,
              int bpp) {
  std::vector<unsigned char> previous(stride, 0);
  std::size_t pos = 0;
  for (int y = 0; y < height; ++y) {
    const int type = raw[pos++];
    unsigned char* line = raw.data() + pos;
    for (std::size_t i = 0; i < stride; ++i) {
      const int left = (i >= static_cast<std::size_t>(bpp)) ? line[i - bpp] : 0;
      const int up = previous[i];
      const int up_left = (i >= static_cast<std::size_t>(bpp)) ? previous[i - bpp] : 0;
      int value = line[i];
      switch (type) {
        case 0: break;
        case 1: value += left; break;
        case 2: value += up; break;
        case 3: value += (left + up) / 2; break;
        case 4: value += paeth(left, up, up_left); break;
        default: throw std::runtime_error("PNG: unknown filter type " +
                                          std::to_string(type));
      }
      line[i] = static_cast<unsigned char>(value & 0xff);
    }
    std::memcpy(previous.data(), line, stride);
    pos += stride;
  }
}

}  // namespace

PngImage read_png_gray(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open PNG: " + path);
  std::vector<unsigned char> file((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());

  static const unsigned char kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  if (file.size() < 8 || std::memcmp(file.data(), kSignature, 8) != 0) {
    throw std::runtime_error("not a PNG: " + path);
  }

  int width = 0, height = 0, bit_depth = 0, colour_type = 0;
  std::vector<unsigned char> idat;
  std::size_t pos = 8;

  while (pos + 8 <= file.size()) {
    const std::uint32_t length = read_be32(file.data() + pos);
    const char* type = reinterpret_cast<const char*>(file.data() + pos + 4);
    const unsigned char* data = file.data() + pos + 8;
    if (pos + 12 + length > file.size()) break;

    if (std::memcmp(type, "IHDR", 4) == 0) {
      width = static_cast<int>(read_be32(data));
      height = static_cast<int>(read_be32(data + 4));
      bit_depth = data[8];
      colour_type = data[9];
      if (data[12] != 0) throw std::runtime_error("PNG: interlaced images unsupported");
    } else if (std::memcmp(type, "IDAT", 4) == 0) {
      idat.insert(idat.end(), data, data + length);
    } else if (std::memcmp(type, "IEND", 4) == 0) {
      break;
    }
    pos += 12 + length;
  }

  if (width <= 0 || height <= 0) throw std::runtime_error("PNG: missing IHDR in " + path);
  if (bit_depth != 8 && bit_depth != 16) {
    throw std::runtime_error("PNG: only 8 and 16-bit samples supported, got " +
                             std::to_string(bit_depth));
  }
  int channels = 0;
  if (colour_type == 0) channels = 1;
  else if (colour_type == 2) channels = 3;
  else throw std::runtime_error("PNG: only greyscale and RGB supported, got colour type " +
                                std::to_string(colour_type));

  const int bytes_per_sample = bit_depth / 8;
  const int bpp = channels * bytes_per_sample;
  const std::size_t stride = static_cast<std::size_t>(width) * bpp;
  const std::size_t expected = (stride + 1) * static_cast<std::size_t>(height);

  std::vector<unsigned char> raw(expected);
  uLongf out_size = static_cast<uLongf>(expected);
  const int status = uncompress(raw.data(), &out_size, idat.data(),
                                static_cast<uLong>(idat.size()));
  if (status != Z_OK || out_size != expected) {
    throw std::runtime_error("PNG: inflate failed on " + path);
  }

  unfilter(raw, height, stride, bpp);

  PngImage image;
  image.width = width;
  image.height = height;
  image.samples.resize(static_cast<std::size_t>(width) * height);

  const float scale = (bit_depth == 16) ? (1.0f / 65535.0f) : (1.0f / 255.0f);
  for (int y = 0; y < height; ++y) {
    const unsigned char* line = raw.data() + (stride + 1) * y + 1;
    for (int x = 0; x < width; ++x) {
      const unsigned char* p = line + static_cast<std::size_t>(x) * bpp;
      const int value = (bit_depth == 16) ? ((p[0] << 8) | p[1]) : p[0];
      image.samples[static_cast<std::size_t>(y) * width + x] =
          static_cast<float>(value) * scale;
    }
  }
  return image;
}

HsiFrame demosaic(const PngImage& raw, int period, int bands,
                  const std::vector<int>& band_order) {
  if (period <= 0) throw std::invalid_argument("demosaic: period must be positive");
  if (bands <= 0 || bands > period * period) {
    throw std::invalid_argument("demosaic: bands must be in 1.." +
                                std::to_string(period * period));
  }
  if (!band_order.empty() && static_cast<int>(band_order.size()) != bands) {
    throw std::invalid_argument("demosaic: band_order must have one entry per band");
  }

  HsiFrame frame;
  frame.shape.width = raw.width / period;
  frame.shape.height = raw.height / period;
  frame.shape.bands = bands;
  frame.interleave = Interleave::Bsq;
  if (!frame.shape.valid()) {
    throw std::runtime_error("demosaic: image is smaller than one mosaic tile");
  }
  frame.data.assign(frame.shape.elements(), 0.0f);

  const std::size_t pixels = frame.shape.pixels();
  for (int b = 0; b < bands; ++b) {
    // Which position in the mosaic tile carries this band.
    const int position = band_order.empty() ? b : band_order[static_cast<std::size_t>(b)];
    const int oy = position / period;
    const int ox = position % period;
    float* plane = frame.data.data() + static_cast<std::size_t>(b) * pixels;
    for (int y = 0; y < frame.shape.height; ++y) {
      const int ry = y * period + oy;
      const float* row = raw.samples.data() + static_cast<std::size_t>(ry) * raw.width;
      float* out = plane + static_cast<std::size_t>(y) * frame.shape.width;
      for (int x = 0; x < frame.shape.width; ++x) {
        out[x] = row[static_cast<std::size_t>(x) * period + ox];
      }
    }
  }
  return frame;
}

namespace {

class HotSource final : public FrameSource {
 public:
  explicit HotSource(const HotOptions& options) : opt_(options) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(opt_.directory)) {
      throw std::runtime_error("HOT source: not a directory: " + opt_.directory);
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(opt_.directory)) {
      if (!entry.is_regular_file()) continue;
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (ext == ".png") paths_.push_back(entry.path().string());
    }
    // Frames are numbered in the filename, so lexicographic order is frame
    // order as long as the numbers are zero-padded, which HOT's are.
    std::sort(paths_.begin(), paths_.end());
    if (paths_.empty()) {
      throw std::runtime_error("HOT source: no PNG frames in " + opt_.directory);
    }

    const HsiFrame first = load(0);
    shape_ = first.shape;

    const int resident = std::max(1, std::min<int>(opt_.max_resident,
                                                   static_cast<int>(paths_.size())));
    cache_.reserve(static_cast<std::size_t>(resident));
    cache_.push_back(first.data);
    for (int i = 1; i < resident; ++i) {
      HsiFrame frame = load(static_cast<std::size_t>(i));
      if (frame.shape != shape_) {
        throw std::runtime_error("HOT source: frame " + paths_[static_cast<std::size_t>(i)] +
                                 " has a different shape than the first");
      }
      cache_.push_back(std::move(frame.data));
    }
  }

  CubeShape shape() const override { return shape_; }
  Interleave interleave() const override { return Interleave::Bsq; }
  std::uint64_t length() const override {
    return opt_.loop ? 0 : static_cast<std::uint64_t>(paths_.size());
  }

  bool read_into(float* dst, FrameMeta* meta) override {
    if (cursor_ >= paths_.size()) {
      if (!opt_.loop) return false;
      cursor_ = 0;
    }
    if (cursor_ < cache_.size()) {
      std::memcpy(dst, cache_[cursor_].data(), cache_[cursor_].size() * sizeof(float));
    } else {
      const HsiFrame frame = load(cursor_);
      std::memcpy(dst, frame.data.data(), frame.data.size() * sizeof(float));
    }
    if (meta) {
      meta->index = emitted_;
      meta->timestamp_s = static_cast<double>(emitted_) / 25.0;  // HOT captures at 25 fps
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
    return "hot(" + std::to_string(paths_.size()) + " frames, " +
           std::to_string(shape_.height) + "x" + std::to_string(shape_.width) +
           "x" + std::to_string(shape_.bands) + ", " +
           std::to_string(cache_.size()) + " resident)";
  }

 private:
  HsiFrame load(std::size_t i) const {
    return demosaic(read_png_gray(paths_[i]), opt_.mosaic_period, opt_.bands,
                    opt_.band_order);
  }

  HotOptions opt_;
  std::vector<std::string> paths_;
  std::vector<std::vector<float>> cache_;
  CubeShape shape_;
  std::size_t cursor_ = 0;
  std::uint64_t emitted_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> make_hot_source(const HotOptions& options) {
  return std::make_unique<HotSource>(options);
}

}  // namespace hsi
