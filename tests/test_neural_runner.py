from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from statarb.backtest.engine import run_backtest
from statarb.config import ResearchConfig
from statarb.neural.config import ModernConfig
from statarb.neural.dataset import build_tape
from statarb.neural.runner import extension_research, run_modern
from statarb.neural.training import decisions, fit_policy
from statarb.panel import ResearchPanel


def synthetic_panel() -> ResearchPanel:
    rng = np.random.default_rng(83)
    dates = pd.bdate_range("2020-01-02", periods=90)
    ids = pd.Index([101, 203, 307, 409], name="security_id")
    etfs = pd.DataFrame(rng.normal(0, 0.008, (90, 2)), index=dates, columns=["SPY", "XLK"])
    noise = rng.normal(0, 0.002, (90, 4))
    returns = pd.DataFrame(
        noise + etfs.to_numpy()[:, [0, 1, 0, 1]] * [0.8, 1.15, 0.95, 1.3],
        index=dates,
        columns=ids,
    )

    return ResearchPanel(
        returns=returns,
        universe=pd.DataFrame(True, index=dates, columns=ids),
        dollar_volume=pd.DataFrame(1e8, index=dates, columns=ids),
        etf_returns=etfs,
        risk_free=pd.Series(0.0, index=dates),
        start=dates[60],
    )


def base_research() -> ResearchConfig:
    return ResearchConfig.model_validate(
        {
            "rebalance_every": 5,
            "model": {
                "kind": "etf",
                "estimation_window": 21,
                "etfs": ["SPY", "XLK"],
                "etf_switch_margin": 0,
            },
            "portfolio": {"max_weight": 0.2, "gross_leverage": 0.8},
            "backtest": {"execution_lag": 2, "min_trade": 0.002},
            "costs": {"impact_coef": 0.03, "spread_model": "liquidity"},
        }
    )


def experiment(panel: ResearchPanel, architecture: str) -> ModernConfig:
    dates = panel.dates
    return ModernConfig.model_validate(
        {
            "architecture": architecture,
            "train_start": dates[28].date(),
            "train_end": dates[55].date(),
            "test_start": dates[60].date(),
            "test_end": dates[79].date(),
            "lookback": 5,
            "hidden_size": 8,
            "max_stocks": 4,
            "training_steps": 2,
            "training_window_days": 20,
            "refit_every": 4,
            "chunk_days": 3,
            "cpu_threads": 1,
            "seed": 123,
            "borrow_bps_annual": 300,
        }
    )


def truncate(panel: ResearchPanel, cutoff: pd.Timestamp) -> ResearchPanel:
    return replace(
        panel,
        **{
            name: getattr(panel, name).loc[:cutoff]
            for name in (
                "returns",
                "universe",
                "dollar_volume",
                "etf_returns",
                "risk_free",
            )
        },
    )


@pytest.mark.parametrize("architecture", ["conv_transformer", "lstm"])
def test_rolling_policy_has_no_future_return_dependency(architecture: str) -> None:
    panel = synthetic_panel()
    base = base_research()
    original = base.model_dump()
    config = experiment(panel, architecture)
    cutoff = panel.dates[70]
    full = run_modern(panel, base, config)
    truncated = run_modern(truncate(panel, cutoff), base, config)
    shocked_returns, shocked_etfs = panel.returns.copy(), panel.etf_returns.copy()
    shocked_returns.loc[shocked_returns.index > cutoff] = 0.7
    shocked_etfs.loc[shocked_etfs.index > cutoff] = -0.4
    shocked = run_modern(
        replace(panel, returns=shocked_returns, etf_returns=shocked_etfs), base, config
    )
    pd.testing.assert_frame_equal(full.targets.loc[:cutoff], truncated.targets, check_exact=True)
    pd.testing.assert_frame_equal(
        full.targets.loc[:cutoff], shocked.targets.loc[:cutoff], check_exact=True
    )
    pd.testing.assert_series_equal(
        full.backtest.net_pnl.loc[:cutoff], truncated.backtest.net_pnl, check_exact=True
    )
    pd.testing.assert_series_equal(
        full.backtest.net_pnl.loc[:cutoff], shocked.backtest.net_pnl.loc[:cutoff], check_exact=True
    )
    assert base.model_dump() == original
    audit = full.fit_audit
    assert len(audit) == 5
    assert (audit.latest_outcome_date <= audit.cutoff).all()
    assert (audit.latest_feature_date <= audit.latest_outcome_date).all()
    assert (audit.cutoff < audit.decision_date).all()
    assert audit.cutoff.iloc[0] <= pd.Timestamp(config.train_end)
    assert (full.targets.abs().sum(axis=1) <= base.portfolio.gross_leverage + 1e-7).all()
    assert (full.targets[panel.ids].abs() <= base.portfolio.max_weight + 1e-7).all().all()
    assert (full.targets.sum(axis=1).abs() < 1e-7).all()
    assert np.isfinite(full.training.loss).all()
    assert np.isfinite(full.backtest.net_pnl).all()


def test_final_cost_accounting_is_the_original_engine_and_base_config_is_unchanged() -> None:
    panel, base = synthetic_panel(), base_research()
    config = experiment(panel, "linear")
    before = base.model_dump()
    result = run_modern(panel, base, config)
    effective = extension_research(base, config)
    returns = pd.concat([panel.returns, panel.etf_returns], axis=1).loc[result.targets.index]
    reference = run_backtest(
        result.targets,
        returns,
        panel.dollar_volume.loc[returns.index],
        effective.backtest,
        effective.costs,
        etfs=set(panel.etf_returns),
    )

    for field in ("gross_pnl", "costs", "net_pnl", "turnover"):
        pd.testing.assert_series_equal(
            getattr(result.backtest, field), getattr(reference, field), check_freq=False
        )

    pd.testing.assert_frame_equal(result.backtest.weights, reference.weights, check_freq=False)
    assert result.backtest.costs.sum() > 0
    assert effective.costs.borrow_bps_annual == 300
    assert effective.backtest.min_trade == 0
    assert base.model_dump() == before


def test_fit_policy_is_deterministic_and_audits_only_completed_outcomes() -> None:
    panel, base = synthetic_panel(), base_research()
    config = experiment(panel, "lstm")
    effective = extension_research(base, config)
    tape = build_tape(panel, effective, max_stocks=config.max_stocks)
    cache = decisions(tape, config.lookback)
    cutoff = pd.Timestamp(config.train_end)
    one, loss_one, audit_one = fit_policy(tape, panel, effective, config, cache, cutoff)
    two, loss_two, audit_two = fit_policy(tape, panel, effective, config, cache, cutoff)

    for name, weights in one.state_dict().items():
        torch.testing.assert_close(weights, two.state_dict()[name], rtol=0, atol=0)

    pd.testing.assert_frame_equal(loss_one, loss_two, check_exact=True)
    assert audit_one == audit_two
    assert audit_one["latest_outcome_date"] <= cutoff
    assert (loss_one.outcome_end <= cutoff).all()


def test_future_only_security_does_not_change_past_portfolios_or_training() -> None:
    panel, base = synthetic_panel(), base_research()
    config = experiment(panel, "conv_transformer")
    cutoff = panel.dates[70]
    newcomer = 999
    returns, universe = panel.returns.copy(), panel.universe.copy()
    volume = panel.dollar_volume.copy()
    returns[newcomer] = np.nan
    universe[newcomer] = False
    volume[newcomer] = np.nan
    after = panel.dates > cutoff
    returns.loc[after, newcomer] = panel.etf_returns.loc[after, "SPY"] + 0.01
    universe.loc[after, newcomer] = True
    volume.loc[after, newcomer] = 2e8
    extended = replace(panel, returns=returns, universe=universe, dollar_volume=volume)
    original = run_modern(panel, base, config)
    with_future_name = run_modern(extended, base, config)
    pd.testing.assert_frame_equal(
        original.targets.loc[:cutoff],
        with_future_name.targets.loc[:cutoff, original.targets.columns],
        check_exact=False,
        atol=1e-8,
        rtol=1e-7,
    )
    assert with_future_name.targets.loc[:cutoff, newcomer].eq(0).all()
    pd.testing.assert_series_equal(
        original.backtest.net_pnl.loc[:cutoff],
        with_future_name.backtest.net_pnl.loc[:cutoff],
        check_exact=False,
        atol=1e-10,
        rtol=1e-7,
    )
    past_refits = original.fit_audit.loc[original.fit_audit.decision_date <= cutoff]
    pd.testing.assert_frame_equal(
        past_refits,
        with_future_name.fit_audit.loc[with_future_name.fit_audit.decision_date <= cutoff],
    )
