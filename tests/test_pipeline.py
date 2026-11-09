"""End-to-end properties of the classical pipeline on the synthetic market"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_contracts import make_context
from statarb.backtest.runner import instrument_returns, rebalance_step, run_strategy
from statarb.config import PortfolioConfig, ResearchConfig
from statarb.models.ou import fit_ou, simulate_ou
from statarb.models.pipeline import RebalanceContext, run_signals
from statarb.models.signals import LONG, SHORT
from statarb.portfolio.construction import target_weights
from statarb.synthetic import synthetic_market

SMALL: dict[str, Any] = {
    "model": {"kind": "etf", "estimation_window": 60, "etfs": ["SPY", "XLE", "XLF", "XLK"]},
    "ou": {"window": 30, "min_kappa": 0.0},
}
CASES: dict[str, dict[str, Any]] = {
    "rule_pca": {"model": {"kind": "pca", "pca_k": 3}},
    "rule_etf_weekly_lag2_band": {
        "rebalance_every": 5,
        "backtest": {"execution_lag": 2, "min_trade": 0.002},
    },
    "calibrated_optimizer": {
        "rebalance_every": 5,
        "portfolio": {"construction": "optimizer"},
        "backtest": {"execution_lag": 2},
        "optimizer": {"horizon": 3, "alpha_calibration": "online", "calib_min_rebalances": 3},
        "turnover": {"smoothing_halflife": 1.0},
        "costs": {"spread_model": "liquidity", "impact_coef": 0.01},
    },
}


def config(**extra: Any) -> ResearchConfig:
    merged = SMALL | extra | {"model": SMALL["model"] | extra.get("model", {})}
    return ResearchConfig.model_validate(merged)


@pytest.fixture(scope="module")
def market():  # type: ignore[no-untyped-def]
    return synthetic_market(n_stocks=40, n_days=420, warmup_days=100, noise_vol=0.0, seed=3)


def test_ou_fit_recovers_mean_reversion_speed() -> None:
    estimates = [fit_ou(simulate_ou(30.0, 0.0, 0.2, 2_000, seed=s)) for s in range(20)]
    kappas = [p.kappa for p in estimates if p is not None]
    assert len(kappas) == 20
    assert np.median(kappas) == pytest.approx(30.0, rel=0.25)


def test_etf_model_recovers_planted_hedges(market) -> None:  # type: ignore[no-untyped-def]
    from statarb.models.residuals import EtfResidualModel

    panel, truth = market.panel, market.truth
    window = panel.returns.iloc[-250:].dropna(axis=1)
    fitted = EtfResidualModel(["SPY", *sorted(set(truth.etf))]).fit(
        window, panel.etf_returns.loc[window.index]
    )
    assert (fitted.etf == truth.etf[fitted.etf.index]).mean() > 0.9
    error = (fitted.beta - truth.beta[fitted.beta.index]).abs()
    assert error.median() < 0.1 and error.max() < 0.3  # se(beta) ~ 0.08 over 250 days


def test_rebalance_context_is_point_in_time(market) -> None:  # type: ignore[no-untyped-def]
    seen: list[RebalanceContext] = []

    def record(ctx: RebalanceContext) -> pd.Series:
        seen.append(ctx)
        return target_weights(ctx, PortfolioConfig())

    result = run_signals(market.panel, config(), on_rebalance=record)
    assert result.targets is not None and len(seen) > 0
    assert list(result.targets.index) == [c.date for c in seen]

    for ctx in seen:
        assert ctx.returns.index.max() == ctx.date
        assert ctx.regressors.index.max() == ctx.date
        assert ctx.positions.equals(result.positions.loc[ctx.date])
        assert set(ctx.ou_params) <= set(ctx.scores.dropna().index)


@pytest.mark.parametrize("case", list(CASES))
def test_results_do_not_depend_on_later_rows(market, case: str) -> None:  # type: ignore[no-untyped-def]
    cfg = config(**CASES[case])
    cutoff = market.panel.dates[300]
    long = run_strategy(market.panel, cfg)
    short = run_strategy(market.panel.truncate(cutoff), cfg)
    assert len(long.targets) > len(short.targets)

    cols = short.targets.columns
    head = long.targets.loc[:cutoff]
    pd.testing.assert_frame_equal(head[cols], short.targets)
    assert (head.drop(columns=cols).fillna(0.0) == 0).all().all()

    if not short.calibration.empty:
        assert (short.calibration["slope"] > 0).any(), "calibration must be exercised"

    for name in ("gross_pnl", "costs", "net_pnl", "turnover"):
        np.testing.assert_allclose(
            getattr(long.backtest, name).loc[:cutoff],
            getattr(short.backtest, name),
            rtol=0,
            atol=1e-15,
            err_msg=name,
        )


def test_default_strategy_charges_costs_and_respects_leverage(market) -> None:  # type: ignore[no-untyped-def]
    cfg = config()
    res = run_strategy(market.panel, cfg)
    bt = res.backtest
    assert bt.net_pnl.index.equals(market.panel.signal_dates)
    assert np.isfinite(bt.net_pnl.to_numpy()).all()
    assert (bt.costs > 0).any()
    assert set(res.targets.columns) <= set(instrument_returns(market.panel).columns)
    held = bt.weights.abs().sum(axis=1)
    assert (held[held > 0] <= cfg.portfolio.gross_leverage + 0.1).all()


def test_profit_comes_from_mean_reverting_names(market) -> None:  # type: ignore[no-untyped-def]
    # a 60 day AR(1) fit is biased towards reversion (Dickey-Fuller), so random walks pass the
    # kappa filter too. The signal still tells them apart and only reversion pays
    cfg = config(ou={"window": 60, "min_kappa": 252 / 30})
    res = run_strategy(market.panel, cfg)
    kappa = res.signals.diagnostics.groupby("security_id")["kappa"].median()
    reverting = market.truth["reverting"]
    assert kappa[reverting[reverting].index].median() > 1.5 * kappa[~reverting].median()

    w = res.backtest.weights
    stocks = [c for c in w.columns if not isinstance(c, str)]
    pnl = (w[stocks] * market.panel.returns.reindex(w.index)[stocks].fillna(0.0)).sum()
    from_reverting, from_random_walks = pnl[reverting[stocks]].sum(), pnl[~reverting[stocks]].sum()
    assert from_reverting > abs(from_random_walks) > 0


def test_rebalance_step_defaults_to_the_rule() -> None:
    def boom(*_: object) -> pd.Series:
        raise AssertionError("not used with the default config")

    ctx = make_context({1: LONG, 2: SHORT, 3: LONG})
    step = rebalance_step(ResearchConfig(), optimizer_fn=boom, smoothing_fn=boom)
    pd.testing.assert_series_equal(step(ctx), target_weights(ctx, PortfolioConfig()))
