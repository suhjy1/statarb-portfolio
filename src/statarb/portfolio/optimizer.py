from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from statarb.config import ResearchConfig
from statarb.models.pipeline import RebalanceContext
from statarb.models.residuals import FittedEtf, FittedModel
from statarb.models.signals import FLAT
from statarb.portfolio.construction import finish_weights
from statarb.portfolio.costs import stock_half_spread_bps


def _active_ids(ctx: RebalanceContext) -> pd.Index:
    ids = ctx.scores.index.intersection(ctx.returns.columns, sort=False)
    return ids[
        (
            np.isfinite(ctx.scores.reindex(ids)) & ctx.positions.reindex(ids).fillna(FLAT).ne(FLAT)
        ).to_numpy()
    ]


@dataclass(frozen=True)
class _AlphaRecord:
    date: pd.Timestamp
    alpha: pd.Series
    fitted: FittedModel


class AlphaCalibrator:
    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.pending: deque[_AlphaRecord] = deque()
        self.history: deque[tuple[float, float]] = deque(
            maxlen=config.optimizer.calib_window_rebalances
        )
        self.slope = 0.0
        self.diagnostics: list[dict[str, object]] = []
        self.last_date: pd.Timestamp | None = None
        self.matured = 0

    def calibrate(self, ctx: RebalanceContext, alpha: pd.Series) -> pd.Series:
        if self.last_date is not None and ctx.date <= self.last_date:
            raise ValueError("calibration dates must be strictly increasing")

        self.last_date = ctx.date
        returns = ctx.returns.loc[: ctx.date]
        dates = pd.DatetimeIndex(returns.index)
        h, lag = self.config.rebalance_every, self.config.backtest.execution_lag
        dropped_ids, dropped_spans = 0, 0

        while self.pending:
            record = self.pending[0]
            pos = dates.get_indexer([record.date])[0]

            if pos < 0:
                self.pending.popleft()
                dropped_spans += 1
                continue

            stop = pos + lag + h

            if stop > len(dates):
                break

            self.pending.popleft()
            span = dates[pos + lag : stop]
            ids = record.alpha.index.intersection(returns.columns, sort=False)
            dropped_ids += len(record.alpha) - len(ids)
            reg = ctx.regressors.reindex(span)
            residuals = record.fitted.residuals(returns.loc[span], reg).reindex(columns=ids)
            realized = residuals.sum(axis=0, min_count=h)
            a = record.alpha.reindex(ids)
            valid = np.isfinite(a) & np.isfinite(realized)
            dropped_ids += int((~valid).sum())

            if not valid.any():
                dropped_spans += 1
                continue

            a, y = a[valid].to_numpy(), realized[valid].to_numpy()
            self.history.append((float(a @ y), float(a @ a)))
            self.matured += 1

        xx = sum(pair[1] for pair in self.history)
        self.slope = (
            float(np.clip(sum(pair[0] for pair in self.history) / xx, 0, 1))
            if len(self.history) >= self.config.optimizer.calib_min_rebalances and xx > 0
            else 0.0
        )
        active = alpha.reindex(_active_ids(ctx)).dropna()

        if not active.empty:
            self.pending.append(_AlphaRecord(ctx.date, active.copy(), ctx.fitted))

        calibrated = alpha * self.slope
        self.diagnostics.append(
            {
                "date": ctx.date,
                "slope": self.slope,
                "matured_window": len(self.history),
                "matured_total": self.matured,
                "active_ids": len(active),
                "dropped_ids": dropped_ids,
                "dropped_spans": dropped_spans,
                "median_raw_alpha_bps": float(active.abs().median() * 1e4),
                "median_calibrated_alpha_bps": float(active.abs().median() * self.slope * 1e4),
            }
        )

        return calibrated


def expected_alpha(ctx: RebalanceContext, config: ResearchConfig) -> pd.Series:
    scores = ctx.scores.astype(float).where(np.isfinite(ctx.scores), 0.0)
    scale = pd.Series(config.optimizer.alpha_scale, index=scores.index)

    for sid, fit in ctx.ou_params.items():
        if sid in scale.index:
            scale[sid] = fit.sigma_eq * (-np.expm1(-fit.kappa * config.rebalance_every / 252))

    return -scores * scale


def residual_covariance(ctx: RebalanceContext) -> pd.DataFrame:
    returns = ctx.returns.loc[: ctx.date]

    if isinstance(ctx.fitted, FittedEtf) and ctx.regressors.empty:
        residuals = returns
    else:
        residuals = ctx.fitted.residuals(returns, ctx.regressors.reindex(returns.index))

    total_var = returns.var(ddof=1)
    fallback = max(float(total_var[total_var > 0].median()), 1e-10)

    if not np.isfinite(fallback):
        fallback = 1e-4

    variance = residuals.reindex(columns=returns.columns).var(ddof=1)
    variance = variance.where(np.isfinite(variance) & (variance > 0), total_var)
    variance = variance.where(np.isfinite(variance) & (variance > 0), fallback).clip(lower=1e-10)

    return pd.DataFrame(np.diag(variance), index=returns.columns, columns=returns.columns)


def optimize_weights(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    previous: pd.Series,
    dollar_volume: pd.Series,
    config: ResearchConfig,
) -> pd.Series:
    if alpha.empty:
        return pd.Series(dtype=float)

    ids = alpha.index
    a = alpha.to_numpy(dtype=float)
    cov = covariance.reindex(index=ids, columns=ids).to_numpy(dtype=float)
    prev = previous.reindex(ids).fillna(0.0).to_numpy(dtype=float)

    if not (np.isfinite(a).all() and np.isfinite(cov).all() and np.isfinite(prev).all()):
        raise ValueError("alpha, covariance and previous weights must be finite and aligned")

    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    diagonal = np.array_equal(cov, np.diag(np.diag(cov)))
    eigen_min = np.diag(cov).min() if diagonal else np.linalg.eigvalsh(cov).min()

    if eigen_min < -1e-12:
        raise ValueError("covariance must be positive semidefinite")

    if not np.any(a) and not np.any(prev):
        return pd.Series(0.0, index=ids)

    opt, port, costs = config.optimizer, config.portfolio, config.costs
    dv = dollar_volume.reindex(ids).astype(float)
    spread = (
        stock_half_spread_bps(dv, costs).to_numpy()
        if costs.spread_model == "liquidity"
        else np.full(len(ids), costs.stock_half_spread_bps)
    ) * 1e-4
    impact = (
        costs.impact_coef
        * np.sqrt(costs.capital / dv.where(np.isfinite(dv) & (dv > 0))).fillna(0.0).to_numpy()
    )
    cap = port.max_weight
    w = cp.Variable((len(ids), opt.horizon))
    constraints = []
    objective = cp.Constant(0.0)

    for stage in range(opt.horizon):
        weights = w[:, stage]
        change = weights - (prev / cap if stage == 0 else w[:, stage - 1])
        size = cp.abs(change)
        risk = (
            cp.sum(cp.multiply(np.maximum(np.diag(cov), 0), cp.square(weights)))
            if diagonal
            else cp.quad_form(weights, cp.psd_wrap(cov))
        )
        cost = (spread * cap) @ size

        if costs.impact_coef > 0:
            impact_size = cp.Variable(len(ids), nonneg=True)
            constraints.append(cp.PowCone3D(impact_size, np.ones(len(ids)), change, 2 / 3))
            cost += (impact * cap**1.5) @ impact_size

        objective += (
            (a * cap * opt.alpha_decay**stage) @ weights
            - opt.risk_aversion * config.rebalance_every * cap**2 * risk
            - opt.cost_aversion * cost
        )
        constraints.extend([cp.abs(weights) <= 1, cp.norm1(weights) <= port.gross_leverage / cap])

        if port.dollar_neutral and not port.hedge:
            constraints.append(cp.sum(weights) == 0)

    scale = max(
        float(np.max(np.abs(a))),
        float(np.max(spread)) * opt.cost_aversion,
        float(np.max(np.diag(cov))) * opt.risk_aversion * config.rebalance_every,
        1e-6,
    )
    problem = cp.Problem(cp.Maximize(objective / (scale * cap)), constraints)
    tolerance = 1e-8 if costs.impact_coef > 0 else 1e-9
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=tolerance, tol_feas=1e-9, tol_gap_rel=tolerance)

    if problem.status != cp.OPTIMAL or w.value is None:
        raise RuntimeError(f"portfolio optimization failed: {problem.status}")

    result = pd.Series(cap * w.value[:, 0], index=ids).clip(-port.max_weight, port.max_weight)
    gross = float(result.abs().sum())

    if gross > port.gross_leverage:
        result *= port.gross_leverage / gross

    return result.where(result.abs() > 1e-9, 0.0)


def optimized_target_weights(
    ctx: RebalanceContext,
    previous: pd.Series,
    config: ResearchConfig,
    *,
    alpha_override: pd.Series | None = None,
) -> pd.Series:
    active = _active_ids(ctx)

    if active.empty:
        return pd.Series(dtype=float)

    alpha = (expected_alpha(ctx, config) if alpha_override is None else alpha_override).reindex(
        active
    )
    cov = residual_covariance(ctx).loc[active, active]
    stocks = optimize_weights(alpha, cov, previous, ctx.dollar_volume, config)
    weights = finish_weights(stocks[stocks != 0], ctx.fitted, config.portfolio)
    stock_mask = weights.index.isin(ctx.returns.columns)
    largest = float(weights[stock_mask].abs().max())

    if largest > config.portfolio.max_weight:
        weights *= config.portfolio.max_weight / largest

    return weights[weights.abs() > 1e-12]
