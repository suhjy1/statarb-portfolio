from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchPanel:
    """Point-in-time market data the models consume, all aligned on one trading calendar

    in the original research build this came from a DuckDB/Parquet store with corporate
    actions, survivorship handling, and provenance. Here I keep it as a plain container so
    the models can run on any source (statarb.synthetic is one example)
    """

    returns: pd.DataFrame  # [date x id] total returns, nan where not listed
    universe: pd.DataFrame  # [date x id] bool, point-in-time eligibility
    dollar_volume: pd.DataFrame  # [date x id]
    etf_returns: pd.DataFrame  # [date x ticker] hedge instruments
    risk_free: pd.Series  # [date] daily risk-free rate
    start: pd.Timestamp  # first date signals may be produced (after warmup)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)

    @property
    def signal_dates(self) -> pd.DatetimeIndex:
        return self.dates[self.dates >= self.start]

    @property
    def ids(self) -> pd.Index:
        return self.returns.columns

    def truncate(self, end: pd.Timestamp) -> ResearchPanel:
        """Drops every row after end, used to test for look-ahead"""
        cut = {
            name: getattr(self, name).loc[:end]
            for name in ("returns", "universe", "dollar_volume", "etf_returns", "risk_free")
        }
        return ResearchPanel(**cut, start=self.start)
