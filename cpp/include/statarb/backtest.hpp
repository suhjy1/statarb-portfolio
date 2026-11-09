// backtest engine
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "statarb/common.hpp"

namespace statarb {

// costs in return units: fractions of capital
struct CostParams {
  double stock_half_spread_bps = 5.0;
  double etf_half_spread_bps = 1.0;
  double impact_coef = 0.0;
  double borrow_bps_annual = 50.0;
  double capital = 1e7;

  // only used when spread_model is liquidity
  bool liquidity_spreads = false;
  double spread_ref_dollar_volume = 1e8;
  double spread_elasticity = 0.5;
  double spread_min_bps = 1.0;
  double spread_max_bps = 100.0;
};

struct BacktestParams {
  int execution_lag = 1;
  double min_trade = 0.0;
};

// rows = days, cols = instruments
struct BacktestOutput {
  Matrix weights;  // held over each day
  std::vector<double> gross_pnl;
  std::vector<double> costs;
  std::vector<double> net_pnl;
  std::vector<double> turnover;
};

BacktestOutput run_backtest(ConstMatrix targets,
                            std::span<const std::int64_t> target_rows,
                            ConstMatrix returns,
                            ConstMatrix dollar_volume,
                            std::span<const std::uint8_t> is_etf,
                            const CostParams& costs,
                            const BacktestParams& params);

}
