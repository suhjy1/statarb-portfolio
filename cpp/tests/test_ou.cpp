// fit_ou_batch tests
#include <gtest/gtest.h>

#include <vector>
#include <cmath>
#include <limits>

#include "statarb/ou.hpp"

TEST(FitOuBatch, ExactStationaryRecursion) {
  const std::vector<double> x{0, 1, 1.5, 1.75, 1.875};
  const auto f = statarb::fit_ou_batch({x.data(), 5, 1}, 1.0 / 252).at(0);
  ASSERT_TRUE(f.valid);
  EXPECT_NEAR(f.kappa, std::log(2.0) * 252, 1e-10);
  EXPECT_NEAR(f.m, 2, 1e-12);
  EXPECT_DOUBLE_EQ(f.sigma, 1e-12);
}

TEST(FitOuBatch, ColumnsAreIndependentAndEmptyBatchWorks) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> x{nan, 0, 1, 1, 1.5, 1.5, 1.75, 1.75, 1.875, 1.875};
  const auto fits = statarb::fit_ou_batch({x.data(), 5, 2}, 1.0 / 252);
  ASSERT_EQ(fits.size(), 2);
  EXPECT_FALSE(fits[0].valid);
  EXPECT_TRUE(fits[1].valid);
  EXPECT_TRUE(statarb::fit_ou_batch({nullptr, 5, 0}, 1.0 / 252).empty());
  EXPECT_FALSE(statarb::fit_ou_batch({x.data(), 2, 1}, 1.0 / 252)[0].valid);
  EXPECT_THROW(statarb::fit_ou_batch({x.data(), 5, 2}, 0), std::invalid_argument);
}

TEST(FitOuBatch, RejectsNonstationaryAndMatchesConstantMinimumNormFit) {
  const std::vector<double> alternating{1, -1, 1, -1, 1};
  EXPECT_FALSE(statarb::fit_ou_batch({alternating.data(), 5, 1}, 1.0 / 252)[0].valid);
  const std::vector<double> constant(10, 2);
  const auto f = statarb::fit_ou_batch({constant.data(), 10, 1}, 1.0 / 252)[0];
  ASSERT_TRUE(f.valid);
  EXPECT_NEAR(f.kappa, -std::log(0.8) * 252, 1e-10);
  EXPECT_NEAR(f.m, 2, 1e-12);
}
