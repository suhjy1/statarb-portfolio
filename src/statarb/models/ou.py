from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class OUParams:
    kappa: float  # mean reversion speed (per year)
    m: float  # long run mean of x
    sigma: float  # diffusion volatility (per sqrt(year))

    @property
    def sigma_eq(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.kappa)

    @property
    def half_life_days(self) -> float:
        return float(np.log(2.0) / self.kappa * TRADING_DAYS)


def fit_ou(x: np.ndarray, dt: float = 1.0 / TRADING_DAYS) -> OUParams | None:
    x = np.asarray(x, dtype=float)

    if len(x) < 3 or not np.all(np.isfinite(x)):
        return None

    x_lag = x[:-1]
    x_next = x[1:]

    # x[t+1] = a + b * x[t] + epsilon
    design = np.column_stack([np.ones(len(x_lag)), x_lag])
    a, b = np.linalg.lstsq(design, x_next, rcond=None)[0]

    # needs to be a stationary, mean reverting AR(1) process
    if not 0.0 < b < 1.0:
        return None

    kappa = -np.log(b) / dt
    m = a / (1.0 - b)

    residuals = x_next - (a + b * x_lag)
    residual_std = float(np.std(residuals, ddof=2))

    sigma = residual_std * np.sqrt(2.0 * kappa / (1.0 - b**2))

    return OUParams(kappa=float(kappa), m=float(m), sigma=float(max(sigma, 1e-12)))


def simulate_ou(
    kappa: float,
    m: float,
    sigma: float,
    n: int,
    *,
    dt: float = 1.0 / TRADING_DAYS,
    x0: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1.0 - b**2) / (2.0 * kappa))
    x = np.empty(n)
    x[0] = m if x0 is None else x0

    for t in range(1, n):
        x[t] = m + b * (x[t - 1] - m) + sd * rng.standard_normal()

    return x
