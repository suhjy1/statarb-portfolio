#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace statarb {

struct NotImplemented : std::logic_error {
  using std::logic_error::logic_error;
};

struct ConstMatrix {
  const double* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;

  double operator()(std::size_t r, std::size_t c) const {
    return data[r * cols + c];
  }
};

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> data;

  Matrix() = default;
  Matrix(std::size_t r, std::size_t c, double fill = 0.0) : rows(r), cols(c), data(r * c, fill) {}

  double& operator()(std::size_t r, std::size_t c) {
    return data[r * cols + c];
  }
  double operator()(std::size_t r, std::size_t c) const {
    return data[r * cols + c];
  }
};

}
