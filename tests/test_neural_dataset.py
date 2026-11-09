from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig
from statarb.models.residuals import FittedEtf, FittedModel
from statarb.neural.dataset import build_tape, features_at
from statarb.panel import ResearchPanel


def world(n: int = 100) -> ResearchPanel:
    rng = np.random.default_rng(204)
    dates = pd.bdate_range("2020-01-02", periods=n)
    market = pd.Series(rng.normal(0, 0.01, n), index=dates)
    ids = pd.Index([10, 2, 7], name="security_id")
    noise = rng.normal(0, 0.002, (n, 3))
    returns = pd.DataFrame(
        noise + market.to_numpy()[:, None] * [0.8, 1.1, 1.2], index=dates, columns=ids
    )

    return ResearchPanel(
        returns=returns,
        universe=pd.DataFrame(True, index=dates, columns=ids),
        dollar_volume=pd.DataFrame(1e8, index=dates, columns=ids),
        etf_returns=market.to_frame("SPY"),
        risk_free=pd.Series(0.0, index=dates),
        start=dates[40],
    )


def config() -> ResearchConfig:
    return ResearchConfig.model_validate(
        {
            "rebalance_every": 5,
            "model": {
                "kind": "etf",
                "estimation_window": 25,
                "etfs": ["SPY"],
                "etf_switch_margin": 0,
            },
        }
    )


def truncate(panel: ResearchPanel, stop: pd.Timestamp) -> ResearchPanel:
    return replace(
        panel,
        **{
            name: getattr(panel, name).loc[:stop]
            for name in (
                "returns",
                "universe",
                "dollar_volume",
                "etf_returns",
                "risk_free",
            )
        },
    )


def test_factor_fit_never_sees_the_observation_it_scores() -> None:
    panel = world()
    windows = []

    class RecordingEtf:
        name = "recording-etf"

        def fit(
            self,
            returns: pd.DataFrame,
            regressors: pd.DataFrame,
            previous: FittedModel | None = None,
        ) -> FittedEtf:
            windows.append(returns.index.copy())
            return FittedEtf(
                pd.Series("SPY", index=returns.columns),
                pd.Series(1.0, index=returns.columns),
                pd.Series(0.0, index=returns.columns),
                True,
            )

    tape = build_tape(panel, config(), model=RecordingEtf())
    assert len(windows) == 15

    for k, rows in enumerate(windows):
        pos = 25 + k * 5
        assert len(rows) == 25
        assert rows.max() == panel.dates[pos - 1]

    date = panel.dates[25]
    pd.testing.assert_series_equal(
        tape.residuals.loc[date],
        panel.returns.loc[date] - panel.etf_returns.loc[date, "SPY"],
        check_names=False,
    )


def test_future_shocks_and_future_only_names_cannot_rewrite_features() -> None:
    panel = world()
    cut = panel.dates[65]
    base = build_tape(truncate(panel, cut), config())
    shocked_returns = panel.returns.copy()
    shocked_returns.loc[shocked_returns.index > cut] *= 100
    shocked_returns[999] = np.nan
    shocked_returns.loc[shocked_returns.index > cut, 999] = 1.0
    universe = panel.universe.reindex(columns=shocked_returns.columns, fill_value=False)
    universe.loc[universe.index > cut, 999] = True
    extended = replace(
        panel,
        returns=shocked_returns,
        universe=universe,
        dollar_volume=panel.dollar_volume.reindex(columns=shocked_returns.columns),
    )
    full = build_tape(extended, config())

    for name in ("residuals", "eligible", "hedge_beta", "hedge_etf"):
        pd.testing.assert_frame_equal(getattr(base, name), getattr(full, name).loc[:cut, panel.ids])

    ids_a, features_a = features_at(base, cut, 10)
    ids_b, features_b = features_at(full, cut, 10)
    pd.testing.assert_index_equal(ids_a, ids_b)
    np.testing.assert_array_equal(features_a, features_b)


def test_missing_day_is_not_zero_filled_in_features() -> None:
    panel = world()
    returns = panel.returns.copy()
    returns.iloc[60, 0] = np.nan
    panel = replace(panel, returns=returns)
    tape = build_tape(panel, config())
    assert np.isnan(tape.residuals.iloc[60, 0])
    ids, x = features_at(tape, panel.dates[65], 10)
    assert 10 not in ids and len(ids) == 2
    assert x.shape == (2, 10, 2) and np.isfinite(x).all()


def test_features_use_only_own_window_and_adv_selection_preserves_history() -> None:
    panel = world()
    tape = build_tape(panel, config(), max_stocks=1)
    assert list(tape.eligible.columns) == [10, 2, 7]
    assert list(tape.eligible.loc[panel.dates[70]][lambda s: s].index) == [2]
    assert tape.residuals.loc[panel.dates[70], [10, 7]].notna().all()
    date = panel.dates[70]
    ids, x = features_at(tape, date, 5)
    raw = tape.residuals.loc[:date, ids].iloc[-5:].to_numpy()
    normalized = raw / np.maximum(raw.std(axis=0), 1e-8)
    np.testing.assert_allclose(x[:, :, 0], normalized.cumsum(axis=0).T, rtol=1e-6)
    np.testing.assert_allclose(x[:, :, 1], normalized.T, rtol=1e-6)
    assert x.dtype == np.float32
    assert features_at(tape, panel.dates[3], 10)[1].shape == (0, 10, 2)


def test_rejects_intraday_and_unsupported_unhedgeable_factor_models() -> None:
    panel = world()

    with pytest.raises(ValueError, match=r"model\.kind"):
        build_tape(panel, ResearchConfig())

    dates = panel.dates + pd.Timedelta(hours=12)

    with pytest.raises(ValueError, match="intraday"):
        build_tape(replace(panel, returns=panel.returns.set_axis(dates)), config())

    with pytest.raises(ValueError, match="max_stocks"):
        build_tape(panel, config(), max_stocks=0)
