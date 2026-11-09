from __future__ import annotations

import pandas as pd

from statarb.config import TurnoverConfig


def smooth_target(target: pd.Series, previous: pd.Series, config: TurnoverConfig) -> pd.Series:
    if config.smoothing_halflife == 0 or previous.empty:
        return target.copy()

    union = target.index.union(previous.index, sort=False)
    keep = 2 ** (-1 / config.smoothing_halflife)

    return keep * previous.reindex(union).fillna(0.0) + (1 - keep) * target.reindex(union).fillna(
        0.0
    )
