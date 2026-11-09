from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statarb.config import ResearchConfig, SignalConfig
from statarb.models import cleaning
from statarb.models.ou import OUParams, fit_ou
from statarb.models.residuals import (
    EtfResidualModel,
    FittedModel,
    PcaResidualModel,
    ResidualModel,
    rolling_windows,
)
from statarb.models.signals import FLAT, s_score, update_positions
from statarb.panel import ResearchPanel

log = logging.getLogger(__name__)

OUFitter = Callable[[np.ndarray], OUParams | None]
OUBatchFitter = Callable[[pd.DataFrame], "dict[object, OUParams | None]"]
Scorer = Callable[[float, OUParams], float]
PositionRule = Callable[[pd.Series, pd.Series, SignalConfig], pd.Series]
DOLLAR_VOLUME_DAYS = 20


@dataclass(frozen=True)
class RebalanceContext:
    date: pd.Timestamp
    positions: pd.Series  # id to long/flat/short, after the position rule
    scores: pd.Series  # id to s score (nan = not scored today)
    fitted: FittedModel  # residual model fitted on the window ending at date
    returns: pd.DataFrame  # estimation window [date x id] as the model saw it (gaps = 0)
    dollar_volume: pd.Series  # id to median daily $ volume over the last 20 days
    regressors: pd.DataFrame = field(default_factory=pd.DataFrame)  # ETF returns over the window
    ou_params: dict[object, OUParams] = field(default_factory=dict)  # scored ids only


OnRebalance = Callable[[RebalanceContext], "pd.Series | None"]


@dataclass(frozen=True)
class SignalResult:
    diagnostics: pd.DataFrame  # date, security_id, s_score, kappa, half_life_days, position
    positions: pd.DataFrame  # [date x id], long/flat/short
    targets: pd.DataFrame | None = None  # [date x instrument] from on_rebalance


def make_model(config: ResearchConfig) -> ResidualModel:
    m = config.model

    if m.kind == "etf":
        return EtfResidualModel(
            m.etfs,
            drop_alpha=m.drop_alpha,
            fallback=m.etf_fallback,
            min_r2_gain=m.etf_min_r2_gain,
            split_halves=m.etf_split_halves,
            switch_margin=m.etf_switch_margin,
        )

    return PcaResidualModel(m.pca_k, drop_alpha=m.drop_alpha)


def prepare_returns(panel: ResearchPanel, config: ResearchConfig) -> pd.DataFrame:
    rets = panel.returns

    if config.cleaning.use_excess_returns:
        rets = cleaning.excess_returns(rets, panel.risk_free)

    rets = cleaning.mask_to_universe(rets, panel.universe)

    if config.cleaning.winsorize_n_mad is not None:
        rets = cleaning.winsorize_mad(rets, config.cleaning.winsorize_n_mad)

    return rets


def prepare_regressors(panel: ResearchPanel, config: ResearchConfig) -> pd.DataFrame:
    if config.cleaning.use_excess_returns:
        return cleaning.excess_returns(panel.etf_returns, panel.risk_free)
    return panel.etf_returns


def run_signals(
    panel: ResearchPanel,
    config: ResearchConfig,
    *,
    model: ResidualModel | None = None,
    ou_fitter: OUFitter = fit_ou,
    ou_batch_fitter: OUBatchFitter | None = None,
    scorer: Scorer = s_score,
    position_rule: PositionRule = update_positions,
    on_rebalance: OnRebalance | None = None,
) -> SignalResult:
    model = model or make_model(config)
    rets = prepare_returns(panel, config)
    regressors = prepare_regressors(panel, config)
    est_len, ou_len = config.model.estimation_window, config.ou.window
    rows: list[dict[str, object]] = []
    positions: dict[pd.Timestamp, pd.Series] = {}
    previous = pd.Series(FLAT, index=panel.ids, dtype="int64")
    last_fit: FittedModel | None = None
    targets: dict[pd.Timestamp, pd.Series] = {}

    for window in rolling_windows(
        panel.dates, est_len, step=config.rebalance_every, first_asof=panel.start
    ):
        in_universe = panel.universe.loc[window.asof]
        est = rets.loc[window.rows, in_universe[in_universe].index]
        est = est[cleaning.eligible(est, config.cleaning.min_obs_fraction)]

        if est.shape[1] == 0:
            continue

        est = cleaning.fill_for_estimation(est)
        reg = regressors.loc[window.rows]
        reg = reg.loc[:, reg.notna().all()]  # e.g. XLC before 2018, drop it rather than zero fill
        fitted = last_fit = model.fit(est, reg, previous=last_fit)
        resid = fitted.residuals(est.iloc[-ou_len:], reg.iloc[-ou_len:])
        cumulative = resid.cumsum()

        if ou_batch_fitter is not None:
            candidates = ou_batch_fitter(cumulative)
        else:
            candidates = {sid: ou_fitter(cumulative[sid].to_numpy()) for sid in cumulative.columns}

        fits: dict[object, OUParams] = {
            sid: params
            for sid, params in candidates.items()
            if params is not None and params.kappa >= config.ou.min_kappa
        }

        if config.signal.center_m and fits:
            m_bar = float(np.mean([p.m for p in fits.values()]))
            fits = {sid: dataclasses.replace(p, m=p.m - m_bar) for sid, p in fits.items()}

        scores = pd.Series(np.nan, index=panel.ids)

        for sid, params in fits.items():
            scores[sid] = scorer(float(cumulative[sid].iloc[-1]), params)
            rows.append(
                {
                    "date": window.asof,
                    "security_id": sid,
                    "s_score": scores[sid],
                    "kappa": params.kappa,
                    "half_life_days": params.half_life_days,
                }
            )

        previous = position_rule(scores, previous, config.signal).reindex(panel.ids).fillna(FLAT)
        positions[window.asof] = previous.astype("int64")

        if on_rebalance is not None:
            recent = panel.dollar_volume.loc[window.rows[-DOLLAR_VOLUME_DAYS:]]
            ctx = RebalanceContext(
                date=window.asof,
                positions=positions[window.asof],
                scores=scores,
                fitted=fitted,
                returns=est,
                dollar_volume=recent.median(axis=0),
                regressors=reg,
                ou_params=fits,
            )
            weights = on_rebalance(ctx)

            if weights is not None:
                targets[window.asof] = weights

    diagnostics = pd.DataFrame(
        rows, columns=["date", "security_id", "s_score", "kappa", "half_life_days"]
    )
    wide = pd.DataFrame(positions).T.reindex(columns=panel.ids).fillna(FLAT).astype("int64")

    if len(diagnostics):
        held = wide.stack().rename("position").reset_index()
        held.columns = ["date", "security_id", "position"]
        diagnostics = diagnostics.merge(held, on=["date", "security_id"], how="left")

    log.info("signals: %d dates, %d scored stock-days", len(wide), len(diagnostics))
    target_frame = None

    if targets:
        target_frame = pd.concat(targets, axis=1, sort=False).T.fillna(0.0)
        target_frame.index = pd.DatetimeIndex(target_frame.index, name="date")

    return SignalResult(diagnostics=diagnostics, positions=wide, targets=target_frame)
