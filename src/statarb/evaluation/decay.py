from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from statarb.config import ResearchConfig
from statarb.evaluation.stats import sharpe_tstat
from statarb.models.pipeline import (
    RebalanceContext,
    prepare_regressors,
    prepare_returns,
    run_signals,
)
from statarb.panel import ResearchPanel


def _ic_summary(values: pd.Series, horizon: int) -> dict[str, float | int]:
    ics = values.dropna()
    lags = max(horizon - 1, int(4 * (len(ics) / 100) ** (2 / 9)))
    return {
        "ic_mean": float(ics.mean()),
        "ic_tstat": sharpe_tstat(ics, lags=lags),
        "n_dates": len(ics),
    }


def fitted_forward_returns(
    ctx: RebalanceContext, returns: pd.DataFrame, regressors: pd.DataFrame, horizon: int
) -> pd.Series:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    after = returns.loc[returns.index > ctx.date].iloc[:horizon]

    if len(after) < horizon:
        return pd.Series(np.nan, index=ctx.returns.columns)

    resid = ctx.fitted.residuals(
        after.reindex(columns=ctx.returns.columns), regressors.reindex(after.index)
    )

    return resid.sum(axis=0, min_count=horizon)


def residual_decay_profile(
    panel: ResearchPanel, config: ResearchConfig, horizons: Sequence[int]
) -> pd.DataFrame:
    returns, regressors = prepare_returns(panel, config), prepare_regressors(panel, config)
    values: dict[int, list[float]] = {h: [] for h in horizons}

    def observe(ctx: RebalanceContext) -> None:
        for horizon in horizons:
            forward = fitted_forward_returns(ctx, returns, regressors, horizon)
            values[horizon].append(information_coefficient(ctx.scores, forward))

    run_signals(panel, config, on_rebalance=observe)

    return pd.DataFrame(
        [_ic_summary(pd.Series(values[h], dtype=float), h) for h in horizons],
        index=pd.Index(horizons, name="horizon"),
    )


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("returns must have sorted, unique dates")

    if (returns < -1).any().any():
        raise ValueError("simple returns cannot be below -1")

    logs = np.log1p(returns.mask(returns.eq(-1), 0.0))
    summed = logs.rolling(horizon, min_periods=horizon).sum()
    loss = returns.eq(-1).rolling(horizon, min_periods=horizon).sum() > 0
    compounded = np.expm1(summed).mask(loss & summed.notna(), -1.0)

    return compounded.shift(-horizon)


def information_coefficient(scores: pd.Series, forward: pd.Series) -> float:
    pairs = pd.concat([scores.rename("score"), forward.rename("forward")], axis=1, join="inner")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()

    if len(pairs) < 3 or (pairs.nunique() < 2).any():
        return float("nan")

    return float(pairs["score"].rank().corr(pairs["forward"].rank()))


def decay_profile(
    diagnostics: pd.DataFrame, returns: pd.DataFrame, horizons: Sequence[int]
) -> pd.DataFrame:
    scores = diagnostics.pivot(index="date", columns="security_id", values="s_score")
    common = scores.index.intersection(returns.index, sort=False)
    rows = []

    for horizon in horizons:
        forward = forward_returns(returns, horizon)
        ics = pd.Series(
            {date: information_coefficient(scores.loc[date], forward.loc[date]) for date in common},
            dtype=float,
        ).dropna()
        rows.append(_ic_summary(ics, horizon))

    return pd.DataFrame(
        rows, index=pd.Index(horizons, name="horizon"), columns=["ic_mean", "ic_tstat", "n_dates"]
    )
