from __future__ import annotations

import numpy as np
import pandas as pd


def excess_returns(returns: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    return returns.sub(risk_free.reindex(returns.index).fillna(0.0), axis=0)


def mask_to_universe(frame: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    mask = universe.reindex(index=frame.index, columns=frame.columns, fill_value=False)
    return frame.where(mask.astype(bool))


def winsorize_mad(frame: pd.DataFrame, n_mad: float) -> pd.DataFrame:
    median = frame.median(axis=1)
    mad = frame.sub(median, axis=0).abs().median(axis=1) * 1.4826
    lower, upper = median - n_mad * mad, median + n_mad * mad

    return frame.clip(lower=lower, upper=upper, axis=0)


def eligible(window: pd.DataFrame, min_obs_fraction: float) -> pd.Index:
    counts = window.notna().sum(axis=0)
    return window.columns[counts >= int(np.ceil(min_obs_fraction * len(window)))]


def fill_for_estimation(window: pd.DataFrame) -> pd.DataFrame:
    return window.fillna(0.0)
