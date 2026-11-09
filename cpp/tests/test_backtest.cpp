// run_backtest tests
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <limits>

#include "statarb/backtest.hpp"

TEST(RunBacktest, DriftAndExecutionLag) {
  const std::vector<double> targets{0.5, -0.5};
  const std::vector<std::int64_t> rows{0};
  const std::vector<double> returns{0, 0, 0.1, 0, 0.1, 0};
  const std::vector<double> dv(6, 1e8);
  const std::vector<std::uint8_t> etf{0, 0};
  const auto out = statarb::run_backtest(
      {targets.data(), 1, 2}, rows, {returns.data(), 3, 2}, {dv.data(), 3, 2}, etf, {}, {});
  EXPECT_DOUBLE_EQ(out.gross_pnl[0], 0);
  EXPECT_NEAR(out.gross_pnl[1], 0.05, 1e-15);
  EXPECT_NEAR(out.weights(2, 0), 0.55 / 1.05, 1e-15);
  EXPECT_NEAR(out.costs[0], 5e-4, 1e-15);
  statarb::BacktestParams lag;
  lag.execution_lag = 2;
  const auto later = statarb::run_backtest(
      {targets.data(), 1, 2}, rows, {returns.data(), 3, 2}, {dv.data(), 3, 2}, etf, {}, lag);
  EXPECT_DOUBLE_EQ(later.gross_pnl[1], 0);
  EXPECT_NEAR(later.gross_pnl[2], 0.05, 1e-15);
}

TEST(RunBacktest, BandAlwaysExitsAndLastDuplicateWins) {
  const std::vector<double> targets{0.2, 0.5, 0.5004, 0};
  const std::vector<std::int64_t> rows{0, 0, 1, 2};
  const std::vector<double> returns(4, 0), dv(4, 1e8);
  const std::vector<std::uint8_t> etf{0};
  statarb::BacktestParams params;
  params.min_trade = 0.001;
  const auto out = statarb::run_backtest(
      {targets.data(), 4, 1}, rows, {returns.data(), 4, 1}, {dv.data(), 4, 1}, etf, {}, params);
  EXPECT_DOUBLE_EQ(out.weights(1, 0), 0.5);
  EXPECT_DOUBLE_EQ(out.weights(2, 0), 0.5);
  EXPECT_DOUBLE_EQ(out.weights(3, 0), 0);
  EXPECT_DOUBLE_EQ(out.turnover[1], 0);
  EXPECT_DOUBLE_EQ(out.turnover[2], 0.5);
}

TEST(RunBacktest, MissingVolumeUsesSpreadCeilingAndEtfSpread) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> targets{0.1, -0.1, 0.2}, returns(3, nan), dv{0, nan, nan};
  const std::vector<std::int64_t> rows{0};
  const std::vector<std::uint8_t> etf{0, 0, 1};
  statarb::CostParams costs;
  costs.liquidity_spreads = true;
  costs.impact_coef = 0.1;
  const auto out = statarb::run_backtest(
      {targets.data(), 1, 3}, rows, {returns.data(), 1, 3}, {dv.data(), 1, 3}, etf, costs, {});
  EXPECT_NEAR(out.costs[0], 0.00202, 1e-15);
  EXPECT_DOUBLE_EQ(out.gross_pnl[0], 0);
}

TEST(RunBacktest, EmptyDaysAndInvalidIndices) {
  const std::vector<double> target{0.5};
  const std::vector<std::int64_t> rows{0}, invalid{-1};
  const std::vector<std::uint8_t> etf{0};
  const auto out = statarb::run_backtest(
      {target.data(), 1, 1}, rows, {nullptr, 0, 1}, {nullptr, 0, 1}, etf, {}, {});
  EXPECT_TRUE(out.weights.data.empty());
  EXPECT_THROW(statarb::run_backtest(
                   {target.data(), 1, 1}, invalid, {nullptr, 0, 1}, {nullptr, 0, 1}, etf, {}, {}),
               std::invalid_argument);
}
