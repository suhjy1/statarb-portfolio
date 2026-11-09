from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_contracts import make_context
from statarb.backtest.engine import run_backtest
from statarb.config import BacktestConfig, CostConfig, PortfolioConfig
from statarb.models.signals import LONG, SHORT
from statarb.portfolio.construction import hedge_weights, stock_weights, target_weights
from statarb.portfolio.costs import trading_cost


def test_cap_redistributes_until_no_name_exceeds_it() -> None:
    ctx = make_context(
        {i: LONG if i % 2 else SHORT for i in range(1, 61)}, vols={1: 0.002, 2: 0.003}
    )
    cfg = PortfolioConfig(sizing="inverse_vol", gross_leverage=2.0, max_weight=0.05)
    w = stock_weights(ctx, cfg)
    assert w.abs().max() <= 0.05 + 1e-12
    assert abs(w.abs().sum() - 2.0) < 1e-9  # 60 names x 0.05 = 3 > 2: target reachable


def test_all_capped_leaves_gross_below_target() -> None:
    ctx = make_context({1: LONG, 2: SHORT, 3: LONG})
    w = stock_weights(ctx, PortfolioConfig(gross_leverage=2.0, max_weight=0.1))
    assert np.allclose(w.abs(), 0.1)


def test_dollar_neutral_book_keeps_the_factor_hedge() -> None:
    ctx = make_context({1: LONG, 2: LONG, 3: LONG, 4: SHORT, 5: SHORT})
    cfg = PortfolioConfig(gross_leverage=2.0, max_weight=1.0)
    w = target_weights(ctx, cfg)
    stocks = w[[i for i in w.index if not isinstance(i, str)]]
    hedge = hedge_weights(stocks, ctx.fitted)
    etf_legs = w[[i for i in w.index if isinstance(i, str)]]
    pd.testing.assert_series_equal(
        etf_legs.sort_index(), hedge.reindex(etf_legs.index).sort_index(), check_names=False
    )
    assert abs(w.sum()) < 1e-9


def test_impact_scales_with_capital() -> None:
    cfg = CostConfig(stock_half_spread_bps=0.0, impact_coef=0.1, capital=1e7)
    trade, dv = pd.Series({1: 0.01}), pd.Series({1: 1e6})
    small = trading_cost(trade, dv, cfg)[1]
    big = trading_cost(trade, dv, cfg.model_copy(update={"capital": 1e9}))[1]
    assert np.isclose(small, 0.1 * 0.01 * np.sqrt(0.01 * 1e7 / 1e6))
    assert np.isclose(big / small, 10.0)  # sqrt(100)


def test_weights_drift_and_rebalance_trades_from_drifted_book() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    r = pd.DataFrame({1: [0.0, 0.10, 0.0, 0.0], 2: [0.0, 0.0, 0.0, 0.0]}, index=dates)
    targets = pd.DataFrame({1: [0.5, 0.5], 2: [-0.5, -0.5]}, index=[dates[0], dates[2]])
    res = run_backtest(
        targets,
        r,
        pd.DataFrame(1e8, index=dates, columns=[1, 2]),
        BacktestConfig(execution_lag=1),
        CostConfig(),
    )
    drifted = res.weights.loc[dates[2], 1]
    assert np.isclose(drifted, 0.55 / 1.05)  # grew with its 10% return
    assert np.isclose(res.turnover.loc[dates[2]], abs(0.5 - 0.55 / 1.05) + abs(-0.5 + 0.5 / 1.05))
    assert res.costs.loc[dates[2]] > 0


def test_no_trade_band_skips_small_changes_but_always_exits() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    r = pd.DataFrame(0.0, index=dates, columns=[1, 2, 3])
    targets = pd.DataFrame(
        {1: [0.5, 0.5004], 2: [-0.5, -0.4], 3: [0.0003, 0.0]}, index=[dates[0], dates[2]]
    )
    dv = pd.DataFrame(1e8, index=dates, columns=[1, 2, 3])
    res = run_backtest(targets, r, dv, BacktestConfig(min_trade=0.001), CostConfig())
    held = res.weights.loc[dates[3]]
    assert held[1] == 0.5  # 4 bp change < band: not traded
    assert held[2] == -0.4  # 10% change: traded
    assert held[3] == 0.0  # exit always trades, even though 3 bp < band


def test_liquidity_spread_model_is_wired_and_constant_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from statarb.portfolio import costs

    trades = pd.Series({1: 0.1, 2: -0.1, "SPY": 0.2})
    dv = pd.Series({1: 1e9, 2: 1e6})
    constant = trading_cost(trades, dv, CostConfig(), {"SPY"})
    assert np.allclose(constant, [0.1 * 5e-4, 0.1 * 5e-4, 0.2 * 1e-4])

    seen: list[pd.Series] = []

    def stub_spread(dollar_volume: pd.Series, config: CostConfig) -> pd.Series:
        # stub: 1 bp for liquid names, 50 bp otherwise
        seen.append(dollar_volume)
        return pd.Series(np.where(dollar_volume > 1e8, 1.0, 50.0), index=dollar_volume.index)

    monkeypatch.setattr(costs, "stock_half_spread_bps", stub_spread)
    liquid = trading_cost(trades, dv, CostConfig(spread_model="liquidity"), {"SPY"})
    assert list(seen[0].index) == [1, 2, "SPY"]
    assert np.allclose(liquid, [0.1 * 1e-4, 0.1 * 50e-4, 0.2 * 1e-4])  # ETF spread unchanged
