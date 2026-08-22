#include "hsi/envi.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hsi {
namespace {

std::string trim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

/// Split a `{a, b, c}` list body on commas and whitespace.
std::vector<std::string> split_list(const std::string& body) {
  std::vector<std::string> out;
  std::string current;
  for (char c : body) {
    if (c == ',' || c == '\n' || c == '\r') {
      const std::string token = trim(current);
      if (!token.empty()) out.push_back(token);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  const std::string token = trim(current);
  if (!token.empty()) out.push_back(token);
  return out;
}

int require_int(const std::map<std::string, std::string>& fields,
                const std::string& key) {
  const auto it = fields.find(key);
  if (it == fields.end()) {
    throw std::runtime_error("ENVI header is missing required field '" + key + "'");
  }
  try {
    return std::stoi(it->second);
  } catch (const std::exception&) {
    throw std::runtime_error("ENVI field '" + key + "' is not an integer: " +
                             it->second);
  }
}

template <typename T>
float decode(const unsigned char* p, bool swap) {
  T value;
  if (swap) {
    unsigned char tmp[sizeof(T)];
    for (std::size_t i = 0; i < sizeof(T); ++i) tmp[i] = p[sizeof(T) - 1 - i];
    std::memcpy(&value, tmp, sizeof(T));
  } else {
    std::memcpy(&value, p, sizeof(T));
  }
  return static_cast<float>(value);
}

bool host_is_little_endian() {
  const std::uint16_t probe = 1;
  unsigned char bytes[2];
  std::memcpy(bytes, &probe, 2);
  return bytes[0] == 1;
}

}  // namespace

std::size_t EnviHeader::sample_size() const {
  switch (data_type) {
    case 1: return 1;               // uint8
    case 2: case 12: return 2;      // int16 / uint16
    case 3: case 13: case 4: return 4;  // int32 / uint32 / float32
    case 5: return 8;               // float64
    default:
      throw std::runtime_error("unsupported ENVI data type " +
                               std::to_string(data_type));
  }
}

EnviHeader parse_envi_header(const std::string& hdr_path) {
  std::ifstream in(hdr_path);
  if (!in) throw std::runtime_error("cannot open ENVI header: " + hdr_path);

  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();

  EnviHeader header;

  // Fields are `key = value`, where value may be a brace-delimited list that
  // spans lines. Scan character-wise rather than line-wise so the wavelength
  // list - which is always wrapped - parses correctly.
  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t eq = text.find('=', pos);
    if (eq == std::string::npos) break;

    // The key runs back to the previous newline.
    const std::size_t line_start = text.find_last_of('\n', eq);
    const std::size_t key_begin = (line_start == std::string::npos) ? 0 : line_start + 1;
    const std::string key = lower(trim(text.substr(key_begin, eq - key_begin)));

    // ';' starts a comment line. Real headers carry licence and attribution
    // blocks up top, and a URL or a sentence in one of them would otherwise
    // become a field whose name happens to collide with something real.
    if (!key.empty() && key[0] == ';') {
      const std::size_t eol = text.find('\n', eq);
      pos = (eol == std::string::npos) ? text.size() : eol + 1;
      continue;
    }

    std::size_t value_begin = text.find_first_not_of(" \t", eq + 1);
    if (value_begin == std::string::npos) break;

    std::string value;
    if (text[value_begin] == '{') {
      const std::size_t close = text.find('}', value_begin);
      if (close == std::string::npos) {
        throw std::runtime_error("unterminated '{' in ENVI header: " + hdr_path);
      }
      value = text.substr(value_begin + 1, close - value_begin - 1);
      pos = close + 1;
    } else {
      const std::size_t eol = text.find('\n', value_begin);
      const std::size_t stop = (eol == std::string::npos) ? text.size() : eol;
      value = text.substr(value_begin, stop - value_begin);
      pos = stop + 1;
    }
    if (!key.empty()) header.fields[key] = trim(value);
  }

  header.shape.width = require_int(header.fields, "samples");
  header.shape.height = require_int(header.fields, "lines");
  header.shape.bands = require_int(header.fields, "bands");
  header.data_type = require_int(header.fields, "data type");

  if (const auto it = header.fields.find("header offset"); it != header.fields.end()) {
    header.header_offset = std::stol(it->second);
  }
  if (const auto it = header.fields.find("byte order"); it != header.fields.end()) {
    header.little_endian = std::stoi(it->second) == 0;
  }
  if (const auto it = header.fields.find("interleave"); it != header.fields.end()) {
    const std::string value = lower(it->second);
    if (value == "bsq") header.interleave = Interleave::Bsq;
    else if (value == "bil") header.interleave = Interleave::Bil;
    else if (value == "bip") header.interleave = Interleave::Bip;
    else throw std::runtime_error("unknown ENVI interleave '" + value + "'");
  }
  if (const auto it = header.fields.find("wavelength"); it != header.fields.end()) {
    for (const std::string& token : split_list(it->second)) {
      try {
        header.wavelengths_nm.push_back(std::stof(token));
      } catch (const std::exception&) {
        // A non-numeric entry means this is not a wavelength list after all.
        header.wavelengths_nm.clear();
        break;
      }
    }
  }
  header.sample_size();  // validates data_type early
  return header;
}

std::string find_envi_data_file(const std::string& hdr_path) {
  std::string stem = hdr_path;
  const std::string suffix = ".hdr";
  if (stem.size() > suffix.size() &&
      lower(stem.substr(stem.size() - suffix.size())) == suffix) {
    stem = stem.substr(0, stem.size() - suffix.size());
  }
  // ENVI never fixed a data-file extension; try the ones in the wild, bare
  // stem first since that is what the standard tooling writes.
  for (const char* ext : {"", ".float", ".dat", ".img", ".bin", ".raw", ".bsq",
                          ".bil", ".bip", ".cube"}) {
    const std::string candidate = stem + ext;
    std::ifstream probe(candidate, std::ios::binary);
    if (probe) return candidate;
  }
  return "";
}

HsiFrame read_envi(const std::string& hdr_path, Interleave to) {
  const EnviHeader header = parse_envi_header(hdr_path);
  const std::string data_path = find_envi_data_file(hdr_path);
  if (data_path.empty()) {
    throw std::runtime_error("no ENVI data file found alongside " + hdr_path);
  }

  const std::size_t sample_size = header.sample_size();
  const std::size_t count = header.shape.elements();
  const std::size_t expected = count * sample_size;

  std::ifstream in(data_path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open ENVI data file: " + data_path);
  in.seekg(0, std::ios::end);
  const auto file_size = static_cast<std::size_t>(in.tellg());
  if (file_size < expected + static_cast<std::size_t>(header.header_offset)) {
    throw std::runtime_error("ENVI data file " + data_path + " is " +
                             std::to_string(file_size) + " bytes, need " +
                             std::to_string(expected + header.header_offset));
  }
  in.seekg(header.header_offset, std::ios::beg);

  std::vector<unsigned char> raw(expected);
  in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(expected));
  if (!in) throw std::runtime_error("short read on ENVI data file: " + data_path);

  const bool swap = header.little_endian != host_is_little_endian();

  // Integers are rescaled to [0,1] by full scale so one detection threshold
  // works across 8/12/16-bit sensors. Floats are assumed to be reflectance
  // already and pass through.
  std::vector<float> values(count);
  float scale = 1.0f;
  switch (header.data_type) {
    case 1: scale = 1.0f / 255.0f; break;
    case 2: scale = 1.0f / 32767.0f; break;
    case 12: scale = 1.0f / 65535.0f; break;
    case 3: case 13: scale = 1.0f / 2147483647.0f; break;
    default: scale = 1.0f; break;
  }
  for (std::size_t i = 0; i < count; ++i) {
    const unsigned char* p = raw.data() + i * sample_size;
    float v = 0.0f;
    switch (header.data_type) {
      case 1: v = static_cast<float>(*p); break;
      case 2: v = decode<std::int16_t>(p, swap); break;
      case 12: v = decode<std::uint16_t>(p, swap); break;
      case 3: v = decode<std::int32_t>(p, swap); break;
      case 13: v = decode<std::uint32_t>(p, swap); break;
      case 4: v = decode<float>(p, swap); break;
      case 5: v = decode<double>(p, swap); break;
      default: break;
    }
    values[i] = v * scale;
  }

  HsiFrame frame;
  frame.shape = header.shape;
  frame.interleave = header.interleave;
  frame.data = std::move(values);
  frame.wavelengths_nm = header.wavelengths_nm;
  convert_frame(frame, to);
  return frame;
}

}  // namespace hsi
