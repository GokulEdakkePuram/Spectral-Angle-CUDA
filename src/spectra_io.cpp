#include "hsi/spectra_io.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hsi {
namespace {

std::vector<std::string> split(const std::string& line, char sep) {
  std::vector<std::string> out;
  std::string field;
  std::istringstream in(line);
  while (std::getline(in, field, sep)) out.push_back(field);
  return out;
}

std::string trim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

}  // namespace

SpectralLibrary read_spectra_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open spectral library: " + path);

  SpectralLibrary library;
  std::vector<float> wavelengths;
  std::string line;
  int line_number = 0;

  while (std::getline(in, line)) {
    ++line_number;
    const std::string trimmed = trim(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    const std::vector<std::string> fields = split(trimmed, ',');
    if (fields.size() < 2) {
      throw std::runtime_error(path + ":" + std::to_string(line_number) +
                               ": expected a name followed by at least one value");
    }

    const std::string name = trim(fields[0]);
    std::vector<float> values;
    values.reserve(fields.size() - 1);
    for (std::size_t i = 1; i < fields.size(); ++i) {
      try {
        values.push_back(std::stof(trim(fields[i])));
      } catch (const std::exception&) {
        throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                 ": '" + fields[i] + "' is not a number");
      }
    }

    if (name == "wavelength" || name == "wavelengths") {
      wavelengths = std::move(values);
      continue;
    }

    Spectrum spectrum;
    spectrum.name = name;
    spectrum.values = std::move(values);
    library.targets.push_back(std::move(spectrum));
  }

  if (library.targets.empty()) {
    throw std::runtime_error(path + ": no target spectra found");
  }
  const int bands = library.targets.front().bands();
  for (const Spectrum& target : library.targets) {
    if (target.bands() != bands) {
      throw std::runtime_error(path + ": target '" + target.name + "' has " +
                               std::to_string(target.bands()) +
                               " bands, expected " + std::to_string(bands));
    }
  }
  if (!wavelengths.empty()) {
    if (static_cast<int>(wavelengths.size()) != bands) {
      throw std::runtime_error(path + ": wavelength row has " +
                               std::to_string(wavelengths.size()) +
                               " entries but the spectra have " +
                               std::to_string(bands) + " bands");
    }
    for (Spectrum& target : library.targets) target.wavelengths_nm = wavelengths;
  }
  return library;
}

void write_spectra_csv(const std::string& path, const SpectralLibrary& library) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write spectral library: " + path);
  out << "# hyperspectral target spectra\n";

  if (!library.targets.empty() && !library.targets.front().wavelengths_nm.empty()) {
    out << "wavelength";
    for (float nm : library.targets.front().wavelengths_nm) out << "," << nm;
    out << "\n";
  }
  for (const Spectrum& target : library.targets) {
    out << target.name;
    for (float v : target.values) out << "," << v;
    out << "\n";
  }
}

}  // namespace hsi
