from __future__ import annotations

import numpy as np
import pandas as pd

from statarb.config import PortfolioConfig
from statarb.models.pipeline import RebalanceContext
from statarb.models.residuals import FittedEtf, FittedModel, FittedPca
from statarb.models.signals import FLAT, LONG, SHORT

NEUTRALISING_ETF = "SPY"


def _cap_weights(w: pd.Series, gross: float, cap: float) -> pd.Series:
    size = w.abs()
    capped = pd.Series(False, index=w.index)
    out = size * gross / size.sum()

    for _ in range(len(w)):
        over = (out > cap + 1e-15) & ~capped

        if not over.any():
            break

        capped |= over
        budget = gross - cap * capped.sum()
        free = size[~capped]
        out[capped] = cap

        if free.empty or budget <= 0:
            break

        out[~capped] = free * budget / free.sum()

    return out * np.sign(w)


def stock_weights(ctx: RebalanceContext, config: PortfolioConfig) -> pd.Series:
    held = ctx.positions[ctx.positions != FLAT]

    if held.empty:
        return pd.Series(dtype=float)

    signs = held.map({LONG: 1.0, SHORT: -1.0}).astype(float)

    if config.sizing == "equal":
        size = pd.Series(1.0, index=held.index)
    else:  # inverse_vol
        vols = ctx.returns[held.index].iloc[-config.vol_window :].std()
        vols = vols.where(vols > 1e-8, vols[vols > 1e-8].median())
        size = 1.0 / vols

    return _cap_weights(size * signs, config.gross_leverage, config.max_weight)


def hedge_weights(stocks: pd.Series, fitted: FittedModel) -> pd.Series:
    if stocks.empty:
        return pd.Series(dtype=float)

    if isinstance(fitted, FittedEtf):
        beta = fitted.beta.reindex(stocks.index)
        etf = fitted.etf.reindex(stocks.index)
        ok = beta.notna() & etf.notna()

        return -(stocks[ok] * beta[ok]).groupby(etf[ok]).sum()

    if isinstance(fitted, FittedPca):
        betas = fitted.betas.reindex(stocks.index).fillna(0.0)
        exposure = betas.T @ stocks  # PC to exposure
        return -(fitted.weights @ exposure)

    return pd.Series(dtype=float)


def _book(stocks: pd.Series, fitted: FittedModel, hedge: bool) -> pd.Series:
    if not hedge or stocks.empty:
        return stocks
    return stocks.add(hedge_weights(stocks, fitted), fill_value=0.0)


def target_weights(ctx: RebalanceContext, config: PortfolioConfig) -> pd.Series:
    stocks = stock_weights(ctx, config)
    return finish_weights(stocks, ctx.fitted, config)


def finish_weights(stocks: pd.Series, fitted: FittedModel, config: PortfolioConfig) -> pd.Series:
    if stocks.empty:
        return pd.Series(dtype=float)

    longs = _book(stocks[stocks > 0], fitted, config.hedge)
    shorts = _book(stocks[stocks < 0], fitted, config.hedge)

    if config.dollar_neutral:
        net_l, net_s = longs.sum(), shorts.sum()

        if net_l * net_s < 0:
            shorts = shorts * (-net_l / net_s)

        weights = longs.add(shorts, fill_value=0.0)
        net = weights.sum()

        if abs(net) > 1e-12:
            weights = weights.add(pd.Series({NEUTRALISING_ETF: -net}), fill_value=0.0)
    else:
        weights = longs.add(shorts, fill_value=0.0)

    gross = weights.abs().sum()

    if gross > config.gross_leverage:
        weights = weights * (config.gross_leverage / gross)

    return weights[weights != 0.0]
