from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from portfolio_contracts import make_context
from statarb.config import ResearchConfig
from statarb.models.ou import OUParams
from statarb.models.residuals import EtfResidualModel, PcaResidualModel
from statarb.models.signals import LONG, SHORT
from statarb.portfolio.optimizer import (
    expected_alpha,
    optimize_weights,
    optimized_target_weights,
    residual_covariance,
)


def test_alpha_uses_annualized_ou_speed_and_rebalance_days() -> None:
    fit = OUParams(kappa=12.6, m=0, sigma=0.2)
    ctx = replace(make_context({1: LONG}), ou_params={1: fit})
    cfg = ResearchConfig(rebalance_every=5)
    assert expected_alpha(ctx, cfg)[1] == pytest.approx(2 * fit.sigma_eq * (1 - np.exp(-0.25)))


def test_etf_variance_uses_regressors_instead_of_total_returns() -> None:
    ctx = make_context({1: LONG, 2: SHORT})
    market = pd.Series(
        np.random.default_rng(4).normal(0, 0.1, len(ctx.returns)), index=ctx.returns.index
    )
    reg = market.to_frame("SPY")
    returns = ctx.returns.add(market, axis=0)
    fitted = EtfResidualModel(["SPY"]).fit(returns, reg)
    ctx = replace(ctx, returns=returns, fitted=fitted, regressors=reg)
    cov = residual_covariance(ctx)
    residual_var = fitted.residuals(returns, reg).var()
    assert np.diag(cov) == pytest.approx(residual_var.to_numpy())
    assert (np.diag(cov) < returns.var().to_numpy() / 5).all()


def test_multiperiod_cost_plan_has_known_interior_solution() -> None:
    cfg = ResearchConfig.model_validate(
        {
            "portfolio": {"hedge": False, "dollar_neutral": False, "max_weight": 1},
            "optimizer": {"horizon": 2, "alpha_decay": 1},
        }
    )
    args = (
        pd.Series({1: 0.002}),
        pd.DataFrame([[0.01]], index=[1], columns=[1]),
        pd.Series(dtype=float),
        pd.Series({1: 1e8}),
    )
    assert optimize_weights(*args, cfg)[1] == pytest.approx(0.0875, abs=1e-6)
    one = cfg.model_copy(update={"optimizer": cfg.optimizer.model_copy(update={"horizon": 1})})
    assert optimize_weights(*args, one)[1] == pytest.approx(0.075, abs=1e-6)


def test_impact_matches_one_asset_first_order_condition() -> None:
    cfg = ResearchConfig.model_validate(
        {
            "portfolio": {"hedge": False, "dollar_neutral": False, "max_weight": 1},
            "costs": {"impact_coef": 0.1, "capital": 1e7},
        }
    )
    w = optimize_weights(
        pd.Series({1: 0.002}),
        pd.DataFrame([[0.01]], index=[1], columns=[1]),
        pd.Series(dtype=float),
        pd.Series({1: 1e8}),
        cfg,
    )[1]
    derivative = 0.002 - 2 * 0.01 * w - 0.0005 - 1.5 * 0.1 * np.sqrt(1e7 / 1e8 * w)
    assert derivative == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("kind", ["etf", "pca"])
def test_final_optimizer_book_respects_cap_and_neutrality(kind: str) -> None:
    ctx = make_context({i: LONG if i % 2 else SHORT for i in range(1, 12)})

    if kind == "pca":
        ctx = replace(ctx, fitted=PcaResidualModel(k=2).fit(ctx.returns, pd.DataFrame()))

    cfg = ResearchConfig.model_validate({"portfolio": {"construction": "optimizer"}})
    w = optimized_target_weights(ctx, pd.Series(dtype=float), cfg)
    assert not w.empty
    assert w.reindex(ctx.returns.columns).fillna(0).abs().max() <= 0.02 + 1e-9
    assert w.abs().sum() <= 2 + 1e-9
    assert abs(w.sum()) < 1e-9
