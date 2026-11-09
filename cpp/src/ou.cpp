#include "statarb/ou.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace statarb {

std::vector<OUFit> fit_ou_batch(ConstMatrix paths, double dt) {
  if (!std::isfinite(dt) || dt <= 0)
    throw std::invalid_argument("dt must be finite and positive");
  std::vector<OUFit> out(paths.cols);
  if (paths.rows < 3 || paths.cols == 0)
    return out;
  if (paths.data == nullptr)
    throw std::invalid_argument("missing paths buffer");
  const auto n = paths.rows - 1;
  const double count = static_cast<double>(n);
  for (std::size_t c = 0; c < paths.cols; ++c) {
    double mx = 0, my = 0;
    bool finite = std::isfinite(paths(0, c));
    for (std::size_t t = 0; t < n; ++t) {
      mx += paths(t, c);
      my += paths(t + 1, c);
      finite = finite && std::isfinite(paths(t + 1, c));
    }
    if (!finite)
      continue;
    mx /= count;
    my /= count;
    double xx = 0, xy = 0;
    for (std::size_t t = 0; t < n; ++t) {
      const double x = paths(t, c) - mx;
      xx += x * x;
      xy += x * (paths(t + 1, c) - my);
    }
    const double large_eigen = count * (1 + mx * mx) + xx;
    const double cutoff = std::numeric_limits<double>::epsilon() * std::max(count, 2.0);
    const bool rank_one = std::sqrt(count * xx) <= cutoff * large_eigen;
    // min norm least squares
    const double b = rank_one ? mx * my / (1 + mx * mx) : xy / xx;
    const double a = rank_one ? my / (1 + mx * mx) : my - b * mx;
    if (!(b > 0 && b < 1))
      continue;
    double error_mean = 0;
    for (std::size_t t = 0; t < n; ++t) {
      error_mean += paths(t + 1, c) - (a + b * paths(t, c));
    }
    error_mean /= count;
    double ss = 0;
    for (std::size_t t = 0; t < n; ++t) {
      const double error = paths(t + 1, c) - (a + b * paths(t, c)) - error_mean;
      ss += error * error;
    }
    const double sd =
        n > 2 ? std::sqrt(ss / (count - 2)) : std::numeric_limits<double>::quiet_NaN();
    const double kappa = -std::log(b) / dt;
    const double sigma = sd * std::sqrt(2 * kappa / (1 - b * b));
    out[c] = {kappa, a / (1 - b), std::max(sigma, 1e-12), true};
  }
  return out;
}

}
