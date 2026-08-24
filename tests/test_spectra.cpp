// The spectral library is the one input an operator edits by hand, so its
// parser has to round-trip cleanly and fail with a message that says where.

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "check.hpp"
#include "hsi/spectra_io.hpp"

using namespace hsi;
namespace fs = std::filesystem;

namespace {

fs::path scratch() {
  const fs::path dir = fs::temp_directory_path() / "hsi_test_spectra";
  fs::create_directories(dir);
  return dir;
}

fs::path write(const std::string& name, const std::string& contents) {
  const fs::path path = scratch() / name;
  std::ofstream out(path);
  out << contents;
  return path;
}

/// Did the call throw, and did the message name the thing that was wrong?
bool throws_mentioning(const std::string& path, const char* needle) {
  try {
    read_spectra_csv(path);
  } catch (const std::exception& e) {
    const std::string message = e.what();
    return message.find(needle) != std::string::npos;
  }
  return false;
}

void test_round_trip() {
  SpectralLibrary original;
  original.targets.push_back({"blood", {0.11f, 0.19f, 0.24f}, {400.0f, 410.0f, 420.0f}});
  original.targets.push_back({"tomato_concentrate", {0.31f, 0.28f, 0.22f},
                              {400.0f, 410.0f, 420.0f}});

  const fs::path path = scratch() / "library.csv";
  write_spectra_csv(path.string(), original);
  const SpectralLibrary reloaded = read_spectra_csv(path.string());

  CHECK(reloaded.size() == 2);
  CHECK(reloaded.bands() == 3);
  if (reloaded.size() == 2) {
    CHECK(reloaded.targets[0].name == "blood");
    CHECK(reloaded.targets[1].name == "tomato_concentrate");
    for (int i = 0; i < 3; ++i) {
      CHECK_CLOSE(reloaded.targets[0].values[i], original.targets[0].values[i], 1e-6);
      CHECK_CLOSE(reloaded.targets[0].wavelengths_nm[i], 400.0 + 10 * i, 1e-3);
    }
  }
}

void test_comments_blanks_and_whitespace() {
  const fs::path path = write("messy.csv", R"(# a hand-edited library

wavelength, 400.0 , 410.0
  blood , 0.5 , 0.25

# trailing comment
ketchup,0.1,0.9
)");
  const SpectralLibrary library = read_spectra_csv(path.string());
  CHECK(library.size() == 2);
  if (library.size() == 2) {
    CHECK(library.targets[0].name == "blood");
    CHECK_CLOSE(library.targets[0].values[1], 0.25, 1e-6);
    CHECK(library.targets[1].name == "ketchup");
  }
}

void test_flatten_and_norms() {
  SpectralLibrary library;
  library.targets.push_back({"a", {3.0f, 4.0f}, {}});   // norm 5
  library.targets.push_back({"b", {0.0f, 2.0f}, {}});   // norm 2

  const std::vector<float> flat = library.flatten();
  CHECK(flat.size() == 4);
  if (flat.size() == 4) {
    // Target-major, which is the layout the kernels index with t * bands + b.
    CHECK_CLOSE(flat[0], 3.0, 1e-6);
    CHECK_CLOSE(flat[2], 0.0, 1e-6);
    CHECK_CLOSE(flat[3], 2.0, 1e-6);
  }
  const std::vector<float> norms = library.norms();
  CHECK_CLOSE(norms[0], 5.0, 1e-5);
  CHECK_CLOSE(norms[1], 2.0, 1e-5);
}

void test_errors_are_specific() {
  // Each of these is a mistake someone will actually make in a text editor,
  // and each message has to say which line and which value.
  const fs::path ragged = write("ragged.csv", "blood,0.1,0.2\nketchup,0.3\n");
  CHECK(throws_mentioning(ragged.string(), "ketchup"));

  const fs::path not_a_number = write("nan.csv", "blood,0.1,oops\n");
  CHECK(throws_mentioning(not_a_number.string(), "oops"));

  const fs::path empty = write("empty.csv", "# nothing but a comment\n");
  CHECK(throws_mentioning(empty.string(), "no target spectra"));

  const fs::path mismatched =
      write("mismatch.csv", "wavelength,400,410,420\nblood,0.1,0.2\n");
  CHECK(throws_mentioning(mismatched.string(), "wavelength row"));

  CHECK(throws_mentioning((scratch() / "does_not_exist.csv").string(), "cannot open"));
}

}  // namespace

int main() {
  test_round_trip();
  test_comments_blanks_and_whitespace();
  test_flatten_and_norms();
  test_errors_are_specific();
  fs::remove_all(scratch());
  return check::finish("spectra");
}
