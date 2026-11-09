from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from statarb.config import CostConfig

TRADING_DAYS = 252


def trading_cost(
    trades: pd.Series,
    dollar_volume: pd.Series,
    config: CostConfig,
    etfs: Collection[str] = (),
) -> pd.Series:
    size = trades.abs().astype(float)
    is_etf = trades.index.isin(list(etfs))
    stock_spread: float | np.ndarray = config.stock_half_spread_bps

    if config.spread_model == "liquidity":
        dv_all = dollar_volume.reindex(trades.index).astype(float)
        stock_spread = stock_half_spread_bps(dv_all, config).to_numpy(dtype=float)

    half_spread = np.where(is_etf, config.etf_half_spread_bps, stock_spread)
    cost = size * half_spread * 1e-4

    if config.impact_coef > 0:
        dv = dollar_volume.reindex(trades.index).astype(float)
        participation = (size * config.capital / dv).where(dv > 0, 0.0).fillna(0.0)
        cost = cost + config.impact_coef * size * np.sqrt(participation)

    return cost


def stock_half_spread_bps(dollar_volume: pd.Series, config: CostConfig) -> pd.Series:
    dv = dollar_volume.astype(float)
    valid = np.isfinite(dv) & (dv > 0)
    spread = (
        config.stock_half_spread_bps
        * (config.spread_ref_dollar_volume / dv.where(valid)) ** config.spread_elasticity
    )

    return spread.where(valid, config.spread_max_bps).clip(
        lower=config.spread_min_bps, upper=config.spread_max_bps
    )


def borrow_cost(weights: pd.Series, config: CostConfig) -> pd.Series:
    shorts = (-weights.astype(float)).clip(lower=0.0)
    return shorts * config.borrow_bps_annual * 1e-4 / TRADING_DAYS
