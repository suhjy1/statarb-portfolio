#include "statarb/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace statarb {

BacktestOutput run_backtest(ConstMatrix targets,
                            std::span<const std::int64_t> target_rows,
                            ConstMatrix returns,
                            ConstMatrix dollar_volume,
                            std::span<const std::uint8_t> is_etf,
                            const CostParams& costs,
                            const BacktestParams& params) {
  const auto n = returns.rows, m = returns.cols;
  if (targets.cols != m || dollar_volume.rows != n || dollar_volume.cols != m ||
      target_rows.size() != targets.rows || is_etf.size() != m) {
    throw std::invalid_argument("inconsistent backtest dimensions");
  }
  if (params.execution_lag < 1 || params.min_trade < 0) {
    throw std::invalid_argument("invalid execution lag or trade band");
  }
  if ((n && m && (!returns.data || !dollar_volume.data)) || (targets.rows && m && !targets.data))
    throw std::invalid_argument("missing input buffer");
  std::vector<std::size_t> trade_on(n, targets.rows);
  const auto offset = static_cast<std::size_t>(params.execution_lag - 1);
  for (std::size_t k = 0; k < targets.rows; ++k) {
    if (target_rows[k] < 0)
      throw std::invalid_argument("negative target row");
    const auto row = static_cast<std::size_t>(target_rows[k]);
    if (row < n && offset < n - row)
      trade_on[row + offset] = k;  // last target wins
  }
  BacktestOutput out{Matrix(n, m),
                     std::vector<double>(n),
                     std::vector<double>(n),
                     std::vector<double>(n),
                     std::vector<double>(n)};
  std::vector<double> w(m, 0);
  for (std::size_t d = 0; d < n; ++d) {
    double gross = 0, borrow = 0;
    for (std::size_t c = 0; c < m; ++c) {
      out.weights(d, c) = w[c];
      const double r = std::isfinite(returns(d, c)) ? returns(d, c) : 0;
      gross += w[c] * r;
      if (w[c] < 0)
        borrow += -w[c] * costs.borrow_bps_annual * 1e-4 / 252;
    }
    out.gross_pnl[d] = gross;
    out.costs[d] = borrow;
    const double growth = 1 + gross;
    for (std::size_t c = 0; c < m; ++c) {
      const double r = std::isfinite(returns(d, c)) ? returns(d, c) : 0;
      w[c] *= 1 + r;
      if (growth > 0)
        w[c] /= growth;
    }
    if (trade_on[d] != targets.rows) {
      for (std::size_t c = 0; c < m; ++c) {
        double target = targets(trade_on[d], c);
        if (std::isnan(target))
          target = 0;
        const double delta = target - w[c];
        if (std::abs(delta) < params.min_trade && target != 0)
          continue;
        const double size = std::abs(delta);
        if (size != 0) {
          const double dv = dollar_volume(d, c);
          double spread = costs.stock_half_spread_bps;
          if (is_etf[c]) {
            spread = costs.etf_half_spread_bps;
          } else if (costs.liquidity_spreads) {
            spread = std::isfinite(dv) && dv > 0 ? costs.stock_half_spread_bps *
                                                       std::pow(costs.spread_ref_dollar_volume / dv,
                                                                costs.spread_elasticity)
                                                 : costs.spread_max_bps;
            spread = std::clamp(spread, costs.spread_min_bps, costs.spread_max_bps);
          }
          double cost = size * spread * 1e-4;
          if (costs.impact_coef > 0 && dv > 0) {
            cost += costs.impact_coef * size * std::sqrt(size * costs.capital / dv);
          }
          out.costs[d] += cost;
          out.turnover[d] += size;
        }
        w[c] = target;
      }
    }
    out.net_pnl[d] = gross - out.costs[d];
  }
  return out;
}

}
