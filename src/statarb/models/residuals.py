from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    asof: pd.Timestamp
    rows: pd.DatetimeIndex


def rolling_windows(
    dates: pd.DatetimeIndex, length: int, *, step: int = 1, first_asof: pd.Timestamp | None = None
) -> Iterator[Window]:
    dates = pd.DatetimeIndex(dates)
    start = length - 1

    if first_asof is not None:
        start = max(start, int(dates.searchsorted(pd.Timestamp(first_asof))))

    for end in range(start, len(dates), step):
        yield Window(asof=dates[end], rows=dates[end - length + 1 : end + 1])


class FittedModel(Protocol):
    def residuals(self, returns: pd.DataFrame, regressors: pd.DataFrame) -> pd.DataFrame: ...


class ResidualModel(Protocol):
    name: str

    def fit(
        self,
        returns: pd.DataFrame,
        regressors: pd.DataFrame,
        previous: FittedModel | None = None,
    ) -> FittedModel: ...


def _nan_residuals(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)


def _r2_and_beta(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yc, xc = y - y.mean(axis=0), x - x.mean(axis=0)
    ss_x, ss_y = (xc**2).sum(axis=0), (yc**2).sum(axis=0)
    cov = xc.T @ yc

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov / ss_x[:, None]
        r2 = cov**2 / (ss_x[:, None] * ss_y[None, :])

    r2[~np.isfinite(r2)] = -np.inf
    alpha = y.mean(axis=0)[None, :] - beta * x.mean(axis=0)[:, None]

    return r2, beta, alpha


@dataclass(frozen=True)
class FittedEtf:
    etf: pd.Series  # id to ticker
    beta: pd.Series  # id to beta
    alpha: pd.Series  # id to intercept (daily)
    drop_alpha: bool

    def residuals(self, returns: pd.DataFrame, regressors: pd.DataFrame) -> pd.DataFrame:
        ids = self.etf.index.intersection(returns.columns)
        ids = ids[self.etf[ids].isin(regressors.columns).to_numpy()]
        hedge = regressors[self.etf[ids].to_numpy()].to_numpy() * self.beta[ids].to_numpy()
        resid = returns[ids].to_numpy() - hedge

        if self.drop_alpha:
            resid = resid - self.alpha[ids].to_numpy()

        out = pd.DataFrame(resid, index=returns.index, columns=ids)

        return out.reindex(columns=returns.columns)


class EtfResidualModel:
    name = "etf"

    def __init__(
        self,
        etfs: Sequence[str],
        *,
        drop_alpha: bool = True,
        fallback: str = "SPY",
        min_r2_gain: float = 0.0,
        split_halves: bool = False,
        switch_margin: float = 0.02,
    ) -> None:
        self.etfs = tuple(etfs)
        self.drop_alpha = drop_alpha
        self.fallback = fallback
        self.min_r2_gain = min_r2_gain
        self.split_halves = split_halves
        self.switch_margin = switch_margin

    def fit(
        self,
        returns: pd.DataFrame,
        regressors: pd.DataFrame,
        previous: FittedModel | None = None,
    ) -> FittedEtf:
        tickers = [t for t in self.etfs if t in regressors.columns]

        if not tickers or returns.shape[1] == 0:
            empty = pd.Series(dtype=float)
            return FittedEtf(pd.Series(dtype=object), empty, empty, self.drop_alpha)

        y = returns.to_numpy(dtype=float)
        x = regressors[tickers].to_numpy(dtype=float)
        r2, beta, alpha = _r2_and_beta(y, x)

        cols = np.arange(y.shape[1])

        if self.fallback in tickers:
            fb = tickers.index(self.fallback)
            others = r2.copy()
            others[fb] = -np.inf
            cand = others.argmax(axis=0)

            with np.errstate(invalid="ignore"):
                accept = r2[cand, cols] - r2[fb] >= self.min_r2_gain
                if self.split_halves:
                    half = len(y) // 2
                    for part in (slice(None, half), slice(half, None)):
                        r2_part = _r2_and_beta(y[part], x[part])[0]
                        accept &= r2_part[cand, cols] - r2_part[fb] >= self.min_r2_gain

            choice = np.where(accept, cand, fb)
        else:
            choice = r2.argmax(axis=0)

        if self.switch_margin > 0 and isinstance(previous, FittedEtf):
            pos = {t: i for i, t in enumerate(tickers)}
            prev = previous.etf.reindex(returns.columns).map(pos)
            prev_idx = prev.fillna(-1).to_numpy(dtype=int)
            has = prev_idx >= 0
            safe = np.where(has, prev_idx, 0)

            with np.errstate(invalid="ignore"):
                keep = has & ~(r2[choice, cols] - r2[safe, cols] >= self.switch_margin)

            keep &= np.isfinite(r2[safe, cols])
            choice = np.where(keep, safe, choice)

        ok = np.isfinite(r2[choice, cols])
        ids, chosen, cols = returns.columns[ok], choice[ok], cols[ok]

        return FittedEtf(
            etf=pd.Series(np.asarray(tickers, dtype=object)[chosen], index=ids),
            beta=pd.Series(beta[chosen, cols], index=ids),
            alpha=pd.Series(alpha[chosen, cols], index=ids),
            drop_alpha=self.drop_alpha,
        )


@dataclass(frozen=True)
class FittedPca:
    weights: pd.DataFrame  # [id x pc]: q_ij = v_ij / sigma_i
    betas: pd.DataFrame  # [id x pc]
    alpha: pd.Series  # id to intercept (daily)
    drop_alpha: bool

    def residuals(self, returns: pd.DataFrame, regressors: pd.DataFrame) -> pd.DataFrame:
        if self.betas.empty:
            return _nan_residuals(returns)

        r = returns.reindex(columns=self.betas.index)
        factors = r.fillna(0.0).to_numpy() @ self.weights.to_numpy()
        resid = r.to_numpy() - factors @ self.betas.to_numpy().T

        if self.drop_alpha:
            resid = resid - self.alpha.to_numpy()

        out = pd.DataFrame(resid, index=returns.index, columns=self.betas.index)

        return out.reindex(columns=returns.columns)


class PcaResidualModel:
    name = "pca"

    def __init__(self, k: int, *, drop_alpha: bool = True) -> None:
        self.k = k
        self.drop_alpha = drop_alpha

    def fit(
        self,
        returns: pd.DataFrame,
        regressors: pd.DataFrame,
        previous: FittedModel | None = None,
    ) -> FittedPca:
        scales = returns.std(axis=0, ddof=0)
        valid = np.isfinite(scales) & (scales > 1e-12)
        r = returns.loc[:, valid]

        if r.shape[1] < 2:
            empty = pd.DataFrame(dtype=float)
            return FittedPca(empty, empty, pd.Series(dtype=float), self.drop_alpha)

        sd = scales[valid]
        z = ((r - r.mean(axis=0)) / sd).to_numpy()
        _, _, vt = np.linalg.svd(z, full_matrices=False)  # rows: eigenvectors of z'z / t
        k = min(self.k, vt.shape[0])
        pcs = [f"PC{i + 1}" for i in range(k)]
        weights = pd.DataFrame(vt[:k].T / sd.to_numpy()[:, None], index=r.columns, columns=pcs)

        factors = r.to_numpy() @ weights.to_numpy()
        design = np.column_stack([np.ones(len(factors)), factors])
        coef = np.linalg.lstsq(design, r.to_numpy(), rcond=None)[0]  # [(1 + k) x n]

        return FittedPca(
            weights=weights,
            betas=pd.DataFrame(coef[1:].T, index=r.columns, columns=pcs),
            alpha=pd.Series(coef[0], index=r.columns),
            drop_alpha=self.drop_alpha,
        )
