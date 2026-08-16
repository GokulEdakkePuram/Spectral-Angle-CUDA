#pragma once

// A test harness small enough to have no dependencies, which matters because
// this has to build on a freshly flashed Jetson as readily as on a laptop.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace check {

inline int& failures() {
  static int count = 0;
  return count;
}

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& detail = {}) {
  if (ok) return;
  ++failures();
  std::fprintf(stderr, "FAIL %s:%d: %s%s%s\n", file, line, expr,
               detail.empty() ? "" : " -- ", detail.c_str());
}

inline bool close(double a, double b, double tolerance) {
  if (std::isnan(a) || std::isnan(b)) return false;
  return std::fabs(a - b) <= tolerance;
}

inline int finish(const char* name) {
  if (failures() == 0) {
    std::printf("ok %s\n", name);
    return 0;
  }
  std::fprintf(stderr, "FAILED %s: %d check(s)\n", name, failures());
  return 1;
}

}  // namespace check

#define CHECK(expr) ::check::report((expr), #expr, __FILE__, __LINE__)
#define CHECK_CLOSE(a, b, tol)                                            \
  ::check::report(::check::close((a), (b), (tol)), #a " ~= " #b, __FILE__, \
                  __LINE__,                                                \
                  std::to_string(static_cast<double>(a)) + " vs " +        \
                      std::to_string(static_cast<double>(b)))
