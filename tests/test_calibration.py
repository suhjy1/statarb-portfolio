from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.models.pipeline import RebalanceContext
from statarb.portfolio.optimizer import AlphaCalibrator


class IdentityResiduals:
    def residuals(self, returns: pd.DataFrame, regressors: pd.DataFrame) -> pd.DataFrame:
        return returns


def context(returns: pd.DataFrame, date: pd.Timestamp) -> RebalanceContext:
    return RebalanceContext(
        date=date,
        positions=pd.Series(1, index=returns.columns),
        scores=pd.Series(-2.0, index=returns.columns),
        fitted=IdentityResiduals(),
        returns=returns,
        dollar_volume=pd.Series(1e8, index=returns.columns),
        regressors=pd.DataFrame(index=returns.index),
    )


def test_recovers_pooled_slope_and_has_exact_zero_startup() -> None:
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2020-01-02", periods=100)
    alpha = pd.DataFrame(rng.normal(0, 0.02, (100, 50)), index=dates)
    realized = 0.1 * alpha.shift(2) + rng.normal(0, 0.0001, (100, 50))
    cfg = ResearchConfig.model_validate(
        {"backtest": {"execution_lag": 2}, "optimizer": {"alpha_calibration": "online"}}
    )
    calibrator = AlphaCalibrator(cfg)

    for k, date in enumerate(dates):
        result = calibrator.calibrate(context(realized, date), alpha.loc[date])
        if k < 27:
            assert calibrator.slope == 0
            assert (result == 0).all()

    assert calibrator.slope == pytest.approx(0.1, abs=0.001)
    assert len(calibrator.history) == 52
    assert calibrator.diagnostics[-1]["matured_total"] == 98


def test_holding_span_excludes_signal_and_lag_days_and_waits_for_last_day() -> None:
    dates = pd.bdate_range("2020-01-02", periods=7)
    returns = pd.DataFrame({1: [10, 20, 0.02, 0.03, 0.05, 100, 200]}, index=dates)
    cfg = ResearchConfig.model_validate(
        {
            "rebalance_every": 3,
            "backtest": {"execution_lag": 2},
            "optimizer": {"calib_min_rebalances": 1},
        }
    )
    calibrator = AlphaCalibrator(cfg)

    for k in range(5):
        a = pd.Series({1: 1.0}) if k == 0 else pd.Series(dtype=float)
        calibrator.calibrate(context(returns, dates[k]), a)

        if k < 4:
            assert calibrator.slope == 0

    assert calibrator.slope == pytest.approx(0.1)

    with pytest.raises(ValueError, match="increasing"):
        calibrator.calibrate(context(returns, dates[4]), pd.Series(dtype=float))


def test_drops_unavailable_ids_and_clips_the_slope() -> None:
    dates = pd.bdate_range("2020-01-02", periods=3)
    cfg = ResearchConfig.model_validate({"optimizer": {"calib_min_rebalances": 1}})
    calibrator = AlphaCalibrator(cfg)
    initial = pd.DataFrame({1: [0], 2: [0]}, index=dates[:1])
    calibrator.calibrate(context(initial, dates[0]), pd.Series({1: 1.0, 2: 1.0}))
    later = pd.DataFrame({1: [0, 2]}, index=dates[:2])
    calibrator.calibrate(context(later, dates[1]), pd.Series(dtype=float))
    assert calibrator.slope == 1
    assert calibrator.diagnostics[-1]["dropped_ids"] == 1
