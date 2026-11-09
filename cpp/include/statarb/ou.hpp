// Ornstein-Uhlenbeck fit
#pragma once

#include <vector>

#include "statarb/common.hpp"

namespace statarb {

struct OUFit {
  double kappa = 0.0;
  double m = 0.0;
  double sigma = 0.0;
  bool valid = false;
};

std::vector<OUFit> fit_ou_batch(ConstMatrix paths, double dt);

}
