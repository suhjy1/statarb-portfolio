from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from statarb.config import ResearchConfig
from statarb.models import cleaning
from statarb.models.pipeline import make_model, prepare_regressors, prepare_returns
from statarb.models.residuals import FittedEtf, ResidualModel
from statarb.panel import ResearchPanel


@dataclass(frozen=True)
class ModernTape:
    dates: pd.DatetimeIndex
    ids: pd.Index
    etfs: pd.Index
    residuals: pd.DataFrame
    eligible: pd.DataFrame
    hedge_beta: pd.DataFrame
    hedge_etf: pd.DataFrame


def _security_key(value: object) -> tuple[int, int | str]:
    if isinstance(value, Integral):
        return 0, int(value)
    return 1, str(value)


def build_tape(
    panel: ResearchPanel,
    research: ResearchConfig,
    *,
    max_stocks: int | None = None,
    model: ResidualModel | None = None,
) -> ModernTape:
    if research.model.kind != "etf":
        raise ValueError("the modern daily tape currently requires model.kind='etf'")

    if max_stocks is not None and (not isinstance(max_stocks, int) or max_stocks < 1):
        raise ValueError("max_stocks must be a positive integer or None")

    dates, ids = panel.dates, panel.ids

    if (
        not dates.is_unique
        or not dates.is_monotonic_increasing
        or dates.hasnans
        or not dates.equals(dates.normalize())
    ):
        raise ValueError(
            "daily tape requires sorted unique midnight dates; intraday bars are not inferred"
        )

    if not ids.is_unique:
        raise ValueError("security ids must be unique")

    returns = prepare_returns(panel, research).replace([np.inf, -np.inf], np.nan)
    regressors = prepare_regressors(panel, research).replace([np.inf, -np.inf], np.nan)
    universe = panel.universe.reindex(index=dates, columns=ids, fill_value=False).fillna(False)
    length = research.model.estimation_window
    minimum = int(np.ceil(length * research.cleaning.min_obs_fraction))
    enough = returns.notna().rolling(length, min_periods=length).sum().shift(1).ge(minimum)
    adv = panel.dollar_volume.reindex(index=dates, columns=ids).rolling(20, min_periods=1).median()
    residuals = pd.DataFrame(np.nan, index=dates, columns=ids)
    hedge_beta = pd.DataFrame(np.nan, index=dates, columns=ids)
    hedge_etf = pd.DataFrame(None, index=dates, columns=ids, dtype=object)
    eligible = pd.DataFrame(False, index=dates, columns=ids)
    estimator = model if model is not None else make_model(research)
    fitted: FittedEtf | None = None
    previous: FittedEtf | None = None

    for pos in range(length, len(dates)):
        date = dates[pos]
        current = universe.loc[date].astype(bool)

        if (pos - length) % research.rebalance_every == 0:
            est = returns.iloc[pos - length : pos].loc[:, current & enough.loc[date]]
            est = est.loc[:, cleaning.eligible(est, research.cleaning.min_obs_fraction)]
            reg = regressors.iloc[pos - length : pos]
            reg = reg.loc[:, reg.notna().all()]
            fitted = None

            if est.shape[1] and reg.shape[1]:
                candidate = estimator.fit(cleaning.fill_for_estimation(est), reg, previous=previous)

                if not isinstance(candidate, FittedEtf):
                    raise ValueError(
                        "the modern daily tape requires fitted ETF beta/ticker mappings"
                    )

                fitted = previous = candidate

        if fitted is None:
            continue

        row = fitted.residuals(returns.iloc[pos : pos + 1], regressors.iloc[pos : pos + 1])
        values = row.iloc[0].reindex(ids)
        beta = fitted.beta.reindex(ids)
        tickers = fitted.etf.reindex(ids)
        known = np.isfinite(beta) & tickers.isin(panel.etf_returns.columns)
        observable = current & known & np.isfinite(values)
        residuals.loc[date] = values.where(observable)
        hedge_beta.loc[date] = beta.where(known)
        hedge_etf.loc[date] = tickers.where(known)
        can_trade = observable & enough.loc[date]

        if max_stocks is not None:
            liquidity = adv.loc[date]
            names = ids[(can_trade & np.isfinite(liquidity) & liquidity.gt(0)).to_numpy()]
            chosen = sorted(names, key=lambda sid: (-float(liquidity[sid]), _security_key(sid)))
            can_trade &= ids.isin(chosen[:max_stocks])

        eligible.loc[date] = can_trade

    return ModernTape(
        dates=dates.copy(),
        ids=ids.copy(),
        etfs=panel.etf_returns.columns.copy(),
        residuals=residuals,
        eligible=eligible,
        hedge_beta=hedge_beta,
        hedge_etf=hedge_etf,
    )


def features_at(
    tape: ModernTape, date: pd.Timestamp, lookback: int = 30
) -> tuple[pd.Index, np.ndarray]:
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")

    timestamp = pd.Timestamp(date)
    pos = tape.dates.get_indexer([timestamp])[0]

    if pos < 0:
        raise KeyError(f"feature date is absent from the tape: {timestamp}")

    empty = np.empty((0, lookback, 2), dtype=np.float32)

    if pos + 1 < lookback:
        return tape.ids[:0], empty

    window = tape.residuals.iloc[pos - lookback + 1 : pos + 1]
    good = tape.eligible.iloc[pos] & np.isfinite(window).all(axis=0)
    ids = tape.ids[good.to_numpy()]

    if len(ids) == 0:
        return ids, empty

    raw = window.loc[:, ids].to_numpy(dtype=float)
    scale = np.maximum(raw.std(axis=0, ddof=0), 1e-8)
    normalized = raw / scale
    features = np.stack((np.cumsum(normalized, axis=0), normalized), axis=-1)

    return ids, np.transpose(features, (1, 0, 2)).astype(np.float32)
