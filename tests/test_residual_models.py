from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from statarb.models.residuals import EtfResidualModel, PcaResidualModel


def _world(
    t: int = 400, n: int = 30, seed: int = 0, drift: float = 0.002
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=t, freq="B")
    spy = rng.normal(0, 0.01, t)
    xlk = spy + rng.normal(0, 0.008, t)
    xlf = spy + rng.normal(0, 0.008, t)
    etfs = pd.DataFrame({"SPY": spy, "XLK": xlk, "XLF": xlf}, index=idx)
    driver = np.column_stack([xlk] * 10 + [xlf] * 10 + [spy] * (n - 20))
    r = driver + rng.normal(0, 0.004, (t, n)) + drift

    return pd.DataFrame(r, index=idx), etfs


@pytest.mark.parametrize("model", [EtfResidualModel(["SPY", "XLK", "XLF"]), PcaResidualModel(3)])
def test_drop_alpha_centres_in_window_residuals(model: EtfResidualModel) -> None:
    r, etfs = _world()
    resid = model.fit(r, etfs).residuals(r, etfs)
    assert resid.mean().abs().max() < 1e-12


def test_keeping_alpha_leaves_the_drift_in() -> None:
    r, etfs = _world(drift=0.002)
    resid = EtfResidualModel(["SPY", "XLK", "XLF"], drop_alpha=False).fit(r, etfs)
    assert resid.residuals(r, etfs).mean().mean() == pytest.approx(0.002, abs=5e-4)


def test_etf_selection_finds_the_true_sector_and_falls_back_to_spy() -> None:
    r, etfs = _world()
    fitted = EtfResidualModel(["SPY", "XLK", "XLF"], min_r2_gain=0.02).fit(r, etfs)
    assert (fitted.etf.iloc[:10] == "XLK").all()
    assert (fitted.etf.iloc[10:20] == "XLF").all()
    assert (fitted.etf.iloc[20:] == "SPY").all()


def test_etf_margin_rejects_a_chance_fit() -> None:
    r, etfs = _world()
    spy_stocks = r.iloc[:, 20:]
    plain = EtfResidualModel(
        ["SPY", "XLK", "XLF"], min_r2_gain=0.0, split_halves=False, switch_margin=0.0
    )
    guarded = EtfResidualModel(["SPY", "XLK", "XLF"], min_r2_gain=0.05)
    assert (guarded.fit(spy_stocks, etfs).etf == "SPY").all()
    assert plain.fit(spy_stocks, etfs).etf.isin(["SPY", "XLK", "XLF"]).all()


def test_etf_split_halves_rejects_an_unstable_fit() -> None:
    r, etfs = _world()
    half = len(r) // 2
    mixed = r.iloc[:, 20].copy()
    mixed.iloc[:half] = etfs["XLK"].iloc[:half] * 3.0
    frame = mixed.to_frame("x")
    assert EtfResidualModel(["SPY", "XLK"], split_halves=False).fit(frame, etfs).etf["x"] == "XLK"
    assert EtfResidualModel(["SPY", "XLK"], split_halves=True).fit(frame, etfs).etf["x"] == "SPY"


def test_etf_without_fallback_uses_best_fit() -> None:
    r, etfs = _world()
    fitted = EtfResidualModel(["XLK", "XLF"], fallback="SPY").fit(r, etfs)
    assert (fitted.etf.iloc[:10] == "XLK").all()


def test_residuals_keep_input_columns_and_nan_unfitted() -> None:
    r, etfs = _world()
    r[99] = 0.0  # zero variance column: cannot be fitted

    for model in (EtfResidualModel(["SPY", "XLK", "XLF"]), PcaResidualModel(3)):
        resid = model.fit(r, etfs).residuals(r, etfs)
        assert list(resid.columns) == list(r.columns)
        assert resid[99].isna().all()
        assert resid.drop(columns=99).notna().all().all()


def test_pca_with_too_few_stocks_returns_nan_instead_of_raising() -> None:
    r, etfs = _world()
    resid = PcaResidualModel(3).fit(r.iloc[:, :1], etfs).residuals(r.iloc[:, :1], etfs)
    assert resid.shape == (len(r), 1)
    assert resid.isna().all().all()


def test_etf_residuals_out_of_sample_use_fitted_betas() -> None:
    r, etfs = _world()
    fitted = EtfResidualModel(["SPY", "XLK", "XLF"]).fit(r.iloc[:300], etfs.iloc[:300])
    new = fitted.residuals(r.iloc[300:], etfs.iloc[300:])
    sid = r.columns[0]
    expected = (
        r[sid].iloc[300:] - fitted.beta[sid] * etfs[fitted.etf[sid]].iloc[300:] - fitted.alpha[sid]
    )
    pd.testing.assert_series_equal(new[sid], expected, check_names=False)


def test_etf_stickiness_keeps_the_previous_etf_unless_clearly_beaten() -> None:
    r, etfs = _world()
    spy_stocks = r.iloc[:, 20:]
    plain = EtfResidualModel(
        ["SPY", "XLK", "XLF"], min_r2_gain=0.0, split_halves=False, switch_margin=0.0
    )
    sticky = EtfResidualModel(
        ["SPY", "XLK", "XLF"], min_r2_gain=0.0, split_halves=False, switch_margin=0.05
    )
    prev = sticky.fit(r.iloc[:, :10].set_axis(spy_stocks.columns, axis=1), etfs)
    assert (prev.etf == "XLK").all()
    assert (sticky.fit(spy_stocks, etfs, previous=prev).etf == "SPY").all()
    tied = (0.5 * etfs["XLK"] + 0.5 * etfs["XLF"]).to_frame("t") + 0.0
    first = plain.fit(tied, etfs).etf["t"]
    other = "XLF" if first == "XLK" else "XLK"
    prev_tied = sticky.fit(tied, etfs)
    forced = dataclasses.replace(prev_tied, etf=pd.Series({"t": other}))
    assert sticky.fit(tied, etfs, previous=forced).etf["t"] == other
    assert plain.fit(tied, etfs, previous=forced).etf["t"] == first


def test_etf_stickiness_ignores_previous_etfs_missing_from_the_window() -> None:
    r, etfs = _world()
    sticky = EtfResidualModel(["SPY", "XLK", "XLF"], switch_margin=0.05)
    prev = sticky.fit(r, etfs)
    prev = dataclasses.replace(prev, etf=prev.etf.where(prev.etf != "XLK", "XLC"))
    fitted = sticky.fit(r, etfs, previous=prev)
    assert (fitted.etf.iloc[:10] == "XLK").all()
