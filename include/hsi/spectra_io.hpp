#pragma once

#include <string>

#include "hsi/types.hpp"

namespace hsi {

/// Read a target spectral library from CSV.
///
/// One optional `wavelength,<nm>,<nm>,...` row, then one row per target:
///
///     wavelength,400.0,410.0,420.0
///     blood,0.11,0.19,0.24
///     tomato,0.31,0.28,0.22
///
/// Plain text on purpose. A spectral library is the one input an operator
/// hand-edits in the field, and it has to survive being opened in whatever
/// happens to be on the machine.
SpectralLibrary read_spectra_csv(const std::string& path);

void write_spectra_csv(const std::string& path, const SpectralLibrary& library);

}  // namespace hsi
