"""Synthetic market with planted mean-reverting residuals

each stock's return is beta * sector_etf + residual. For a known fraction of stocks the
cumulative residual follows an Ornstein-Uhlenbeck process, and for the rest it is a random walk

since I know the ground truth, this doubles as a test bed. The residual model should recover
the hedge ETF and beta, the OU fit should recover the mean-reversion speed, and the signal
should only trade the reverting names
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.models.ou import TRADING_DAYS
from statarb.panel import ResearchPanel

SECTOR_ETFS = ("XLE", "XLF", "XLK", "XLV", "XLY")


@dataclass(frozen=True)
class SyntheticMarket:
    panel: ResearchPanel
    truth: pd.DataFrame  # [id] ETF, beta, kappa (per year, nan = random walk), reverting


def synthetic_market(
    n_stocks: int = 150,
    n_days: int = 900,
    *,
    warmup_days: int = 320,
    reverting_share: float = 0.6,
    kappa_range: tuple[float, float] = (15.0, 60.0),
    residual_vol: float = 0.25,
    noise_vol: float = 0.3,
    seed: int = 0,
) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_days)
    dt = 1.0 / TRADING_DAYS

    spy = rng.normal(0.0003, 0.010, n_days)
    sector = {
        etf: rng.uniform(0.8, 1.2) * spy + rng.normal(0.0, 0.007, n_days) for etf in SECTOR_ETFS
    }
    etf_returns = pd.DataFrame({"SPY": spy, **sector}, index=dates)

    ids = pd.Index(range(10_001, 10_001 + n_stocks), name="security_id")
    etf = rng.choice(SECTOR_ETFS, n_stocks)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    reverting = rng.random(n_stocks) < reverting_share
    kappa = np.where(reverting, rng.uniform(*kappa_range, n_stocks), np.nan)

    residual = np.empty((n_days, n_stocks))
    step_sd = residual_vol * np.sqrt(dt)

    for j in range(n_stocks):
        shocks = rng.normal(0.0, step_sd, n_days)
        if not reverting[j]:
            residual[:, j] = shocks
            continue
        # exact OU discretisation of the cumulative residual X, so residual return = dX
        b = np.exp(-kappa[j] * dt)
        sd = residual_vol * np.sqrt((1 - b**2) / (2 * kappa[j]))
        x = np.zeros(n_days + 1)
        for t in range(n_days):
            x[t + 1] = b * x[t] + sd * rng.standard_normal()
        residual[:, j] = np.diff(x)

    # non-reverting idiosyncratic noise dilutes the planted edge for every stock
    residual += rng.normal(0.0, noise_vol * np.sqrt(dt), (n_days, n_stocks))
    systematic = etf_returns[etf].to_numpy() * beta
    returns = pd.DataFrame(systematic + residual, index=dates, columns=ids)

    # a few late listings so the universe is not static
    universe = pd.DataFrame(True, index=dates, columns=ids)
    for j in rng.choice(n_stocks, n_stocks // 15, replace=False):
        listed = int(rng.integers(50, n_days // 2))
        universe.iloc[:listed, j] = False
        returns.iloc[:listed, j] = np.nan

    adv = np.exp(rng.normal(np.log(5e7), 0.8, n_stocks))
    dollar_volume = pd.DataFrame(
        adv * np.exp(rng.normal(0.0, 0.2, (n_days, n_stocks))), index=dates, columns=ids
    ).where(universe)

    panel = ResearchPanel(
        returns=returns,
        universe=universe,
        dollar_volume=dollar_volume,
        etf_returns=etf_returns,
        risk_free=pd.Series(0.02 / TRADING_DAYS, index=dates),
        start=dates[warmup_days],
    )
    truth = pd.DataFrame({"etf": etf, "beta": beta, "kappa": kappa, "reverting": reverting}, ids)

    return SyntheticMarket(panel=panel, truth=truth)
