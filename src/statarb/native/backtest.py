from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from statarb.backtest.engine import BacktestResult, BorrowCostFn, TradingCostFn
from statarb.config import BacktestConfig, CostConfig
from statarb.native.ou import cpp_module
from statarb.portfolio.costs import borrow_cost, trading_cost


def cost_params(costs: CostConfig) -> Any:
    p = cpp_module().CostParams()

    for name in (
        "stock_half_spread_bps",
        "etf_half_spread_bps",
        "impact_coef",
        "borrow_bps_annual",
        "capital",
        "spread_ref_dollar_volume",
        "spread_elasticity",
        "spread_min_bps",
        "spread_max_bps",
    ):
        setattr(p, name, float(getattr(costs, name)))

    p.liquidity_spreads = costs.spread_model == "liquidity"

    return p


def backtest_params(config: BacktestConfig) -> Any:
    p = cpp_module().BacktestParams()
    p.execution_lag = int(config.execution_lag)
    p.min_trade = float(config.min_trade)

    return p


def run_backtest_cpp(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    config: BacktestConfig,
    costs: CostConfig,
    *,
    etfs: Collection[str] = (),
    trading_cost_fn: TradingCostFn = trading_cost,
    borrow_cost_fn: BorrowCostFn = borrow_cost,
) -> BacktestResult:
    if trading_cost_fn is not trading_cost or borrow_cost_fn is not borrow_cost:
        raise ValueError("the C++ engine has its own cost model; use the Python reference")

    dates = pd.DatetimeIndex(returns.index)
    # never sort, the columns mix int ids and str tickers
    cols = returns.columns.union(targets.columns, sort=False)
    tgt = targets.reindex(columns=cols).fillna(0.0)
    rows = dates.searchsorted(pd.DatetimeIndex(tgt.index)).astype(np.int64)
    dv = dollar_volume.reindex(index=dates, columns=cols).to_numpy(dtype=float)
    out = cpp_module().run_backtest(
        np.ascontiguousarray(tgt.to_numpy(dtype=float)),
        np.ascontiguousarray(rows),
        np.ascontiguousarray(returns.reindex(columns=cols).to_numpy(dtype=float)),
        np.ascontiguousarray(dv),
        np.asarray(cols.isin(list(etfs)), dtype=np.uint8),
        cost_params(costs),
        backtest_params(config),
    )
    names = ("gross_pnl", "costs", "net_pnl", "turnover")
    series = {k: pd.Series(out[k], index=dates) for k in names}

    return BacktestResult(
        weights=pd.DataFrame(out["weights"], index=dates, columns=cols),
        gross_pnl=series["gross_pnl"],
        costs=series["costs"],
        net_pnl=series["net_pnl"],
        turnover=series["turnover"],
    )
