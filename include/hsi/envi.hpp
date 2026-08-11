#pragma once

#include <map>
#include <string>
#include <vector>

#include "hsi/types.hpp"

namespace hsi {

/// Parsed contents of an ENVI `.hdr` sidecar.
struct EnviHeader {
  CubeShape shape;
  Interleave interleave = Interleave::Bsq;
  int data_type = 4;        ///< ENVI code: 1=u8 2=i16 3=i32 4=f32 5=f64 12=u16 13=u32
  long header_offset = 0;   ///< bytes to skip at the start of the data file
  bool little_endian = true;
  std::vector<float> wavelengths_nm;
  std::map<std::string, std::string> fields;  ///< everything, verbatim

  /// Bytes per sample implied by `data_type`.
  std::size_t sample_size() const;
};

/// Parse an ENVI header. Throws std::runtime_error on malformed input or an
/// unsupported data type.
EnviHeader parse_envi_header(const std::string& hdr_path);

/// Locate the binary cube that belongs to `hdr_path`.
///
/// ENVI never standardised this: the data may sit next to the header with no
/// extension at all, or with any of a handful of conventional ones. Returns an
/// empty string if nothing plausible is found.
std::string find_envi_data_file(const std::string& hdr_path);

/// Read an ENVI cube into a float32 frame with the requested interleave.
///
/// Integer sample types are converted to float and divided by their full-scale
/// value so a detection threshold means the same thing regardless of sensor
/// bit depth. Float sample types are passed through untouched.
HsiFrame read_envi(const std::string& hdr_path,
                   Interleave to = Interleave::Bsq);

}  // namespace hsi
