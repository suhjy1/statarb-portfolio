from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config import BacktestConfig, CostConfig
from statarb.portfolio.costs import borrow_cost, trading_cost

TradingCostFn = Callable[[pd.Series, pd.Series, CostConfig, Collection[str]], pd.Series]
BorrowCostFn = Callable[[pd.Series, CostConfig], pd.Series]


@dataclass(frozen=True)
class BacktestResult:
    weights: pd.DataFrame
    gross_pnl: pd.Series  # [date] sum(weights x returns)
    costs: pd.Series  # [date] trading + borrow, >= 0
    net_pnl: pd.Series
    turnover: pd.Series  # [date] sum |trades| executed that day


def run_backtest(
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
    dates = pd.DatetimeIndex(returns.index)
    cols = returns.columns.union(targets.columns, sort=False)
    r = returns.reindex(columns=cols).to_numpy(dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    tgt = targets.reindex(columns=cols).fillna(0.0)
    dv = dollar_volume.reindex(index=dates, columns=cols)

    # trade day for each target row, as a row index into dates
    trade_on: dict[int, np.ndarray] = {}

    for t, row in tgt.iterrows():
        k = int(dates.searchsorted(pd.Timestamp(t))) + config.execution_lag - 1
        if k < len(dates):
            trade_on[k] = row.to_numpy(dtype=float)

    n, m = r.shape
    w = np.zeros(m)
    held = np.zeros((n, m))
    gross, cost, turnover = np.zeros(n), np.zeros(n), np.zeros(n)

    for d in range(n):
        held[d] = w
        gross[d] = w @ r[d]
        cost[d] = float(borrow_cost_fn(pd.Series(w, index=cols), costs).sum()) if w.any() else 0.0
        growth = 1.0 + gross[d]
        w = w * (1.0 + r[d]) / growth if growth > 0 else w * (1.0 + r[d])

        if d in trade_on:
            target = trade_on[d]
            delta = target - w
            skip = (np.abs(delta) < config.min_trade) & (target != 0.0)
            new_w = np.where(skip, w, target)
            trades = pd.Series(new_w - w, index=cols)
            trades = trades[trades != 0.0]

            if not trades.empty:
                cost[d] += float(trading_cost_fn(trades, dv.iloc[d], costs, etfs).sum())
                turnover[d] = float(trades.abs().sum())

            w = new_w

    gross_s = pd.Series(gross, index=dates)
    cost_s = pd.Series(cost, index=dates)

    return BacktestResult(
        weights=pd.DataFrame(held, index=dates, columns=cols),
        gross_pnl=gross_s,
        costs=cost_s,
        net_pnl=gross_s - cost_s,
        turnover=pd.Series(turnover, index=dates),
    )


def summarize(result: BacktestResult) -> pd.Series:
    pnl = result.net_pnl
    ann_return = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    cum = pnl.cumsum()
    peak = cum.cummax().clip(lower=0.0)

    return pd.Series(
        {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
            "max_drawdown": float(min((cum - peak).min(), 0.0)),
            "avg_turnover": result.turnover.mean(),
            "avg_gross": result.weights.abs().sum(axis=1).mean(),
            "cost_drag": result.costs.mean() * 252,
        }
    )
