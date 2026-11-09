from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from parity.harness import assert_parity
from statarb.backtest.engine import BacktestResult, run_backtest
from statarb.config import BacktestConfig, CostConfig

if os.environ.get("STATARB_REQUIRE_CPP") == "1":
    import statarb_cpp  # noqa: F401
else:
    pytest.importorskip("statarb_cpp", reason="build the C++ kernels first (see README)")

from statarb.native.backtest import run_backtest_cpp

ETFS = ["SPY", "XLK"]


def world(seed: int = 0, n: int = 260, m: int = 40) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    cols: list[object] = [*range(m), *ETFS]
    r = pd.DataFrame(rng.normal(0.0, 0.02, (n, len(cols))), index=dates, columns=cols)
    r = r.mask(rng.random(r.shape) < 0.01)  # nan returns while held
    when = dates[::5]
    t = pd.DataFrame(rng.normal(0.0, 0.02, (len(when), len(cols))), index=when, columns=cols)
    t = t.where(rng.random(t.shape) < 0.6, 0.0)  # entries and exits
    dv = pd.DataFrame(rng.uniform(1e6, 1e9, (n, m)), index=dates, columns=list(range(m)))
    dv = dv.mask(rng.random(dv.shape) < 0.02)  # missing $ volume, ETFs have none

    return {"targets": t, "returns": r, "dollar_volume": dv}


CASES = {
    "lag1": (BacktestConfig(execution_lag=1), CostConfig()),
    "lag2_band": (BacktestConfig(execution_lag=2, min_trade=0.002), CostConfig()),
    "impact": (BacktestConfig(execution_lag=2), CostConfig(impact_coef=0.1)),
    "liquidity": (BacktestConfig(), CostConfig(spread_model="liquidity", impact_coef=0.1)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("field", ["gross_pnl", "costs", "net_pnl", "turnover", "weights"])
def test_backtest_matches_python(case: str, field: str) -> None:
    config, costs = CASES[case]

    def run(fn: object) -> BacktestResult:
        return fn(**world(), config=config, costs=costs, etfs=set(ETFS))  # type: ignore[operator]

    assert_parity(
        lambda: getattr(run(run_backtest), field),
        lambda: getattr(run(run_backtest_cpp), field),
        {},
        atol=1e-12,
    )


def test_custom_cost_functions_are_refused() -> None:
    w = world()
    with pytest.raises(ValueError, match="own cost model"):
        run_backtest_cpp(
            **w,
            config=BacktestConfig(),
            costs=CostConfig(),  # type: ignore[arg-type]
            borrow_cost_fn=lambda weights, cfg: weights * 0.0,
        )


@pytest.mark.parametrize("mode", ["empty", "no_targets", "duplicate", "total_loss"])
def test_boundary_cases(mode: str) -> None:
    dates = pd.bdate_range("2020-01-02", periods=5)
    returns = pd.DataFrame({1: [0, -1, np.nan, 0.1, 0.1]}, index=dates)
    targets = pd.DataFrame({1: [1, 0.2, 0]}, index=dates[[0, 0, 3]])

    if mode == "empty":
        returns = returns.iloc[:0]
    elif mode == "no_targets":
        targets = targets.iloc[:0]
    elif mode == "total_loss":
        targets = pd.DataFrame({1: [1.0]}, index=dates[:1])

    dv = pd.DataFrame({1: [np.nan, 0, np.inf, -1, 1e8]}, index=dates)
    cfg, costs = BacktestConfig(), CostConfig(spread_model="liquidity", impact_coef=0.1)
    ref = run_backtest(targets, returns, dv, cfg, costs)
    actual = run_backtest_cpp(targets, returns, dv, cfg, costs)

    for name in ("weights", "gross_pnl", "costs", "net_pnl", "turnover"):
        np.testing.assert_allclose(getattr(ref, name), getattr(actual, name), atol=1e-12)
