from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def _observations(pnl: pd.Series) -> np.ndarray:
    values = pnl.dropna().to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("P&L must be finite after dropping NaN")

    return values


def sharpe_tstat(pnl: pd.Series, lags: int | None = None) -> float:
    x = _observations(pnl)
    n = len(x)

    if lags is not None and (not isinstance(lags, int) or lags < 0):
        raise ValueError("lags must be a nonnegative integer")

    if n < 2 or np.ptp(x) == 0:
        return float("nan")

    lag_count = min(lags if lags is not None else int(4 * (n / 100) ** (2 / 9)), n - 1)
    mean = x.mean()
    centered = x - mean
    variance = float(centered @ centered)

    for lag in range(1, lag_count + 1):
        variance += 2 * (1 - lag / (lag_count + 1)) * float(centered[lag:] @ centered[:-lag])

    variance /= n - 1

    return float(mean / np.sqrt(variance / n)) if variance > 0 else float("nan")


def probabilistic_sharpe_ratio(pnl: pd.Series, sr_benchmark: float = 0.0) -> float:
    if not np.isfinite(sr_benchmark):
        raise ValueError("sr_benchmark must be finite and per period")

    x = _observations(pnl)
    n = len(x)

    if n < 3 or np.ptp(x) == 0:
        return float("nan")

    centered = x - x.mean()
    m2 = float(np.mean(centered**2))

    if m2 <= 0:
        return float("nan")

    sr = float(x.mean() / x.std(ddof=1))
    skew = float(np.mean(centered**3) / m2**1.5)
    kurtosis = float(np.mean(centered**4) / m2**2)
    variance = 1 - skew * sr + (kurtosis - 1) * sr**2 / 4

    if variance <= 0:
        return float("nan")

    return NormalDist().cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(variance))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    if not isinstance(n_trials, int) or n_trials < 2:
        raise ValueError("n_trials must be an integer >= 2")

    if not np.isfinite(sr_variance) or sr_variance < 0:
        raise ValueError("sr_variance must be finite and nonnegative")

    normal = NormalDist()

    return float(
        np.sqrt(sr_variance)
        * (
            -(1 - np.euler_gamma) * normal.inv_cdf(1 / n_trials)
            - np.euler_gamma * normal.inv_cdf(1 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(pnl: pd.Series, n_trials: int, sr_variance: float) -> float:
    return probabilistic_sharpe_ratio(pnl, expected_max_sharpe(n_trials, sr_variance))
