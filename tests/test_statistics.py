from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from portfolio_contracts import make_context
from statarb.evaluation.decay import (
    fitted_forward_returns,
    forward_returns,
    information_coefficient,
)
from statarb.evaluation.stats import expected_max_sharpe, probabilistic_sharpe_ratio, sharpe_tstat
from statarb.evaluation.walk_forward import WalkForwardConfig, make_folds, select_variant


def test_hac_matches_statsmodels_intercept_regression() -> None:
    pnl = pd.Series(np.random.default_rng(7).normal(0.0003, 0.01, 500))
    reference = sm.OLS(pnl, np.ones((len(pnl), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 8, "use_correction": True}
    )
    assert sharpe_tstat(pnl, 8) == pytest.approx(reference.tvalues.iloc[0], rel=1e-12)


def test_expected_max_matches_paper_daily_units() -> None:
    assert expected_max_sharpe(100, 0.002) == pytest.approx(0.1132, abs=5e-5)
    assert expected_max_sharpe(100, 0) == 0

    with pytest.raises(ValueError):
        expected_max_sharpe(1, 0.002)

    with pytest.raises(ValueError):
        expected_max_sharpe(10, -1)


def test_psr_matches_balanced_two_point_distribution() -> None:
    pnl = pd.Series(np.tile([-0.009, 0.011], 50))
    from statistics import NormalDist

    sr = pnl.mean() / pnl.std(ddof=1)
    expected = NormalDist().cdf(sr * np.sqrt(len(pnl) - 1))
    assert probabilistic_sharpe_ratio(pnl) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("values", [[], [0.01], [0.0] * 20, [0.01] * 20])
def test_degenerate_statistics_are_undefined(values: list[float]) -> None:
    pnl = pd.Series(values, dtype=float)
    assert np.isnan(sharpe_tstat(pnl))
    assert np.isnan(probabilistic_sharpe_ratio(pnl))


def test_forward_compounds_and_preserves_gaps_and_total_losses() -> None:
    dates = pd.bdate_range("2024-01-02", periods=7)
    returns = pd.DataFrame({1: [0, 0.1, 0.2, np.nan, 0.1, -1, 0.2]}, index=dates)
    forward = forward_returns(returns, 2)[1]
    assert forward.iloc[0] == pytest.approx(0.32)
    assert forward.iloc[1:3].isna().all()
    assert forward.iloc[3] == -1
    assert forward.iloc[4] == -1
    assert forward.iloc[-2:].isna().all()
    pd.testing.assert_frame_equal(forward_returns(returns, 10), returns * np.nan)

    with pytest.raises(ValueError):
        forward_returns(returns, 0)

    assert np.isnan(information_coefficient(pd.Series([1, 1, 1]), pd.Series([1, 2, 3])))


def test_folds_keep_short_tail_and_selection_ties_use_name() -> None:
    dates = pd.bdate_range("2020-01-02", periods=23)
    cfg = WalkForwardConfig(train_days=10, test_days=5, embargo_days=2, scheme="rolling")
    folds = make_folds(dates, cfg)
    assert [f.test_end for f in folds] == [dates[16], dates[21], dates[22]]
    assert make_folds(dates[:12], cfg) == []

    with pytest.raises(ValueError):
        make_folds(dates[::-1], cfg)

    daily = pd.DataFrame(
        {"net_pnl": [-0.01, -0.02], "gross_pnl": [0.01, 0.02], "gross_exposure": 1.0}
    )
    assert select_variant({"z": daily, "a": daily}) == "a"

    with pytest.raises(ValueError):
        select_variant({})

    with pytest.raises(ValueError, match="gross_exposure"):
        select_variant({"a": daily.drop(columns="gross_exposure")})


def test_residual_forward_returns_use_the_frozen_hedge_and_require_complete_span() -> None:
    from dataclasses import replace

    ctx = make_context({1: 1, 2: -1, 3: 1})
    dates = pd.bdate_range(ctx.date, periods=4)
    regressors = pd.DataFrame(
        {"XLK": [10, 0.01, 0.02, 0.03], "XLF": [10, 0.02, 0.03, 0.04]}, index=dates
    )
    returns = pd.DataFrame(
        {1: [100, 0.02, 0.03, 0.04], 2: [100, 0.03, 0.04, 0.05], 3: [100, 0.04, 0.05, 0.06]},
        index=dates,
    )
    expected = ctx.fitted.residuals(returns.iloc[1:3], regressors.iloc[1:3]).sum()
    actual = fitted_forward_returns(ctx, returns, regressors, 2)
    pd.testing.assert_series_equal(actual, expected)
    missing = returns.copy()
    missing.iloc[2, 0] = np.nan
    assert np.isnan(fitted_forward_returns(ctx, missing, regressors, 2)[1])
    assert fitted_forward_returns(replace(ctx, date=dates[-1]), returns, regressors, 1).isna().all()
