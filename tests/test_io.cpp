// Round-trips through the two on-disk formats the pipeline ingests: ENVI
// cubes, and the snapshot-mosaic PNGs that HOT-style hyperspectral video ships
// as. Both are written here and read back, so the parsers are checked against
// bytes rather than against themselves.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <zlib.h>

#include "check.hpp"
#include "hsi/envi.hpp"
#include "hsi/hot_loader.hpp"
#include "hsi/types.hpp"

using namespace hsi;
namespace fs = std::filesystem;

namespace {

fs::path scratch() {
  const fs::path dir = fs::temp_directory_path() / "hsi_test_io";
  fs::create_directories(dir);
  return dir;
}

void write_be32(std::vector<unsigned char>& out, std::uint32_t v) {
  out.push_back(static_cast<unsigned char>(v >> 24));
  out.push_back(static_cast<unsigned char>(v >> 16));
  out.push_back(static_cast<unsigned char>(v >> 8));
  out.push_back(static_cast<unsigned char>(v));
}

void write_chunk(std::ofstream& out, const char* type,
                 const std::vector<unsigned char>& data) {
  std::vector<unsigned char> header;
  write_be32(header, static_cast<std::uint32_t>(data.size()));
  out.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));

  std::vector<unsigned char> body(type, type + 4);
  body.insert(body.end(), data.begin(), data.end());
  out.write(reinterpret_cast<const char*>(body.data()),
            static_cast<std::streamsize>(body.size()));

  std::vector<unsigned char> trailer;
  write_be32(trailer, static_cast<std::uint32_t>(
                          crc32(crc32(0L, nullptr, 0), body.data(),
                                static_cast<uInt>(body.size()))));
  out.write(reinterpret_cast<const char*>(trailer.data()),
            static_cast<std::streamsize>(trailer.size()));
}

/// Write a 16-bit greyscale PNG. `filter` selects the per-scanline filter type
/// so the decoder's unfilter paths get exercised rather than just type 0.
void write_png16(const fs::path& path, int width, int height,
                 const std::vector<std::uint16_t>& samples, int filter) {
  std::ofstream out(path, std::ios::binary);
  const unsigned char signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  out.write(reinterpret_cast<const char*>(signature), 8);

  std::vector<unsigned char> ihdr;
  write_be32(ihdr, static_cast<std::uint32_t>(width));
  write_be32(ihdr, static_cast<std::uint32_t>(height));
  ihdr.push_back(16);  // bit depth
  ihdr.push_back(0);   // greyscale
  ihdr.push_back(0);   // deflate
  ihdr.push_back(0);   // adaptive filtering
  ihdr.push_back(0);   // no interlace
  write_chunk(out, "IHDR", ihdr);

  const std::size_t stride = static_cast<std::size_t>(width) * 2;
  std::vector<unsigned char> raw;
  std::vector<unsigned char> previous(stride, 0);
  for (int y = 0; y < height; ++y) {
    std::vector<unsigned char> line(stride);
    for (int x = 0; x < width; ++x) {
      const std::uint16_t v = samples[static_cast<std::size_t>(y) * width + x];
      line[static_cast<std::size_t>(x) * 2] = static_cast<unsigned char>(v >> 8);
      line[static_cast<std::size_t>(x) * 2 + 1] = static_cast<unsigned char>(v & 0xff);
    }
    std::vector<unsigned char> encoded(stride);
    for (std::size_t i = 0; i < stride; ++i) {
      const int left = (i >= 2) ? line[i - 2] : 0;
      const int up = previous[i];
      int predictor = 0;
      if (filter == 1) predictor = left;
      else if (filter == 2) predictor = up;
      else if (filter == 3) predictor = (left + up) / 2;
      encoded[i] = static_cast<unsigned char>((line[i] - predictor) & 0xff);
    }
    raw.push_back(static_cast<unsigned char>(filter));
    raw.insert(raw.end(), encoded.begin(), encoded.end());
    previous = line;
  }

  uLongf bound = compressBound(static_cast<uLong>(raw.size()));
  std::vector<unsigned char> idat(bound);
  compress(idat.data(), &bound, raw.data(), static_cast<uLong>(raw.size()));
  idat.resize(bound);
  write_chunk(out, "IDAT", idat);
  write_chunk(out, "IEND", {});
}

void test_envi_round_trip() {
  const fs::path dir = scratch();
  const fs::path hdr = dir / "cube.hdr";
  const fs::path dat = dir / "cube.float";

  const int width = 5, height = 3, bands = 4;
  CubeShape shape;
  shape.width = width;
  shape.height = height;
  shape.bands = bands;

  // BIL on disk, which is what line-scan sensors write and the layout a
  // line-oriented header parser most often gets wrong.
  std::vector<float> disk(shape.elements());
  for (int y = 0; y < height; ++y)
    for (int b = 0; b < bands; ++b)
      for (int x = 0; x < width; ++x)
        disk[(static_cast<std::size_t>(y) * bands + b) * width + x] =
            static_cast<float>(b * 100 + y * 10 + x);

  {
    std::ofstream out(dat, std::ios::binary);
    out.write(reinterpret_cast<const char*>(disk.data()),
              static_cast<std::streamsize>(disk.size() * sizeof(float)));
  }
  {
    std::ofstream out(hdr);
    // Shaped like a real HyperBlood header: attribution comments above the
    // fields, an empty braced description, and a wrapped wavelength list.
    out << "ENVI\n"
        << ";HSI test image, see http://example.org/licenses/by/4.0/\n"
        << ";samples = 9999 - a comment that looks exactly like a field\n"
        << "description = {\n  a wrapped description with = signs in it\n}\n"
        << "samples = " << width << "\nlines = " << height << "\n"
        << "bands = " << bands << "\nheader offset = 0\n"
        << "file type = ENVI Standard\ndata type = 4\ninterleave = bil\n"
        << "byte order = 0\n"
        // Deliberately wrapped: a line-based parser truncates this to two
        // entries and the wavelengths silently go missing.
        << "wavelength = {\n 400.0, 450.0,\n 500.0, 550.0}\n";
  }

  const EnviHeader header = parse_envi_header(hdr.string());
  CHECK(header.shape == shape);
  CHECK(header.interleave == Interleave::Bil);
  CHECK(header.data_type == 4);
  // The comment lines must not have become fields. Without the guard,
  // ";samples = 9999" parses as a field named ";samples" - harmless here, but
  // it means every sentence with an equals sign in an attribution block ends
  // up in the map, and one of them eventually collides with something real.
  CHECK(header.fields.count("samples") == 1);
  CHECK(header.fields.count(";samples") == 0);
  for (const auto& entry : header.fields) CHECK(entry.first[0] != ';');

  CHECK(header.wavelengths_nm.size() == 4);
  if (header.wavelengths_nm.size() == 4) {
    CHECK_CLOSE(header.wavelengths_nm[0], 400.0, 1e-3);
    CHECK_CLOSE(header.wavelengths_nm[3], 550.0, 1e-3);
  }
  CHECK(find_envi_data_file(hdr.string()) == dat.string());

  const HsiFrame frame = read_envi(hdr.string(), Interleave::Bsq);
  CHECK(frame.interleave == Interleave::Bsq);
  CHECK(frame.shape == shape);
  for (int b = 0; b < bands; ++b)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        CHECK_CLOSE(frame.at(b, y, x), b * 100 + y * 10 + x, 1e-4);
}

void test_envi_uint16_is_rescaled() {
  const fs::path dir = scratch();
  const fs::path hdr = dir / "u16.hdr";
  const fs::path dat = dir / "u16.float";

  const std::vector<std::uint16_t> values{0, 32767, 65535, 16383};
  {
    std::ofstream out(dat, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * 2));
  }
  {
    std::ofstream out(hdr);
    out << "ENVI\nsamples = 2\nlines = 2\nbands = 1\ndata type = 12\n"
        << "interleave = bsq\nbyte order = 0\n";
  }

  const HsiFrame frame = read_envi(hdr.string(), Interleave::Bsq);
  CHECK_CLOSE(frame.data[0], 0.0, 1e-6);
  CHECK_CLOSE(frame.data[2], 1.0, 1e-6);
  CHECK_CLOSE(frame.data[1], 0.5, 1e-4);
}

void test_interleave_conversions_round_trip() {
  CubeShape shape;
  shape.width = 7;
  shape.height = 5;
  shape.bands = 3;

  std::vector<float> original(shape.elements());
  for (std::size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<float>(i) * 0.25f;
  }

  for (Interleave from : {Interleave::Bsq, Interleave::Bil, Interleave::Bip}) {
    for (Interleave to : {Interleave::Bsq, Interleave::Bil, Interleave::Bip}) {
      std::vector<float> forward(shape.elements());
      std::vector<float> back(shape.elements());
      convert_interleave(original.data(), forward.data(), shape, from, to);
      convert_interleave(forward.data(), back.data(), shape, to, from);
      bool same = true;
      for (std::size_t i = 0; i < original.size(); ++i) {
        if (original[i] != back[i]) same = false;
      }
      CHECK(same);
    }
  }
}

void test_png_decode_and_demosaic() {
  const fs::path dir = scratch();
  const int period = 4, bands = 16;
  const int out_w = 6, out_h = 5;
  const int raw_w = out_w * period, raw_h = out_h * period;

  // Plant a value that encodes exactly which band and which spatial pixel a
  // sample belongs to, so a transposed or off-by-one demosaic cannot pass.
  std::vector<std::uint16_t> raw(static_cast<std::size_t>(raw_w) * raw_h);
  auto expected = [&](int b, int y, int x) {
    return static_cast<std::uint16_t>(b * 4001 + y * 61 + x * 7);
  };
  for (int y = 0; y < raw_h; ++y) {
    for (int x = 0; x < raw_w; ++x) {
      const int band = (y % period) * period + (x % period);
      raw[static_cast<std::size_t>(y) * raw_w + x] =
          expected(band, y / period, x / period);
    }
  }

  for (int filter : {0, 1, 2, 3}) {
    const fs::path png = dir / ("mosaic_f" + std::to_string(filter) + ".png");
    write_png16(png, raw_w, raw_h, raw, filter);

    const PngImage image = read_png_gray(png.string());
    CHECK(image.width == raw_w);
    CHECK(image.height == raw_h);

    const HsiFrame frame = demosaic(image, period, bands);
    CHECK(frame.shape.width == out_w);
    CHECK(frame.shape.height == out_h);
    CHECK(frame.shape.bands == bands);
    CHECK(frame.interleave == Interleave::Bsq);

    bool all_match = true;
    for (int b = 0; b < bands; ++b) {
      for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
          const double want = expected(b, y, x) / 65535.0;
          if (!check::close(frame.at(b, y, x), want, 1e-5)) all_match = false;
        }
      }
    }
    CHECK(all_match);
  }
}

void test_demosaic_band_order_permutes() {
  const fs::path dir = scratch();
  const int period = 2, bands = 4;
  const std::vector<std::uint16_t> raw{10, 20, 30, 40};  // 2x2 -> one pixel, 4 bands
  const fs::path png = dir / "tiny.png";
  write_png16(png, 2, 2, raw, 0);

  const PngImage image = read_png_gray(png.string());

  const HsiFrame plain = demosaic(image, period, bands);
  CHECK_CLOSE(plain.at(0, 0, 0), 10.0 / 65535.0, 1e-6);
  CHECK_CLOSE(plain.at(3, 0, 0), 40.0 / 65535.0, 1e-6);

  // Reverse the mosaic-position-to-wavelength mapping.
  const HsiFrame permuted = demosaic(image, period, bands, {3, 2, 1, 0});
  CHECK_CLOSE(permuted.at(0, 0, 0), 40.0 / 65535.0, 1e-6);
  CHECK_CLOSE(permuted.at(3, 0, 0), 10.0 / 65535.0, 1e-6);
}

}  // namespace

int main() {
  test_envi_round_trip();
  test_envi_uint16_is_rescaled();
  test_interleave_conversions_round_trip();
  test_png_decode_and_demosaic();
  test_demosaic_band_order_permutes();
  fs::remove_all(scratch());
  return check::finish("io");
}
