from __future__ import annotations

import pandas as pd

from statarb.config import SignalConfig
from statarb.models.ou import OUParams

LONG, FLAT, SHORT = 1, 0, -1


def s_score(x_last: float, params: OUParams) -> float:
    return float((x_last - params.m) / max(params.sigma_eq, 1e-12))


def update_positions(scores: pd.Series, previous: pd.Series, config: SignalConfig) -> pd.Series:
    scores = scores.astype(float)
    previous = previous.reindex(scores.index).fillna(FLAT).astype("int64")
    positions = previous.copy()

    is_flat = previous == FLAT
    positions[is_flat & (scores < config.open_long)] = LONG
    positions[is_flat & (scores > config.open_short)] = SHORT

    # exit when the score crosses back through the close threshold
    positions[(previous == LONG) & (scores > config.close_long)] = FLAT
    positions[(previous == SHORT) & (scores < config.close_short)] = FLAT

    positions[scores.isna()] = FLAT

    return positions
