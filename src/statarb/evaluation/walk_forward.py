from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scheme: Literal["expanding", "rolling"] = "expanding"
    train_days: int = Field(default=756, gt=0, description="Trading days (expanding: minimum).")
    test_days: int = Field(default=252, gt=0, description="Trading days per test window.")
    embargo_days: int = Field(default=0, ge=0, description="Trading days between train and test.")


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(dates: pd.DatetimeIndex, config: WalkForwardConfig) -> list[Fold]:
    if not dates.is_monotonic_increasing or not dates.is_unique or dates.hasnans:
        raise ValueError("dates must be sorted, unique and nonmissing")

    folds = []

    for start in range(config.train_days + config.embargo_days, len(dates), config.test_days):
        train_stop = start - config.embargo_days
        train_start = 0 if config.scheme == "expanding" else train_stop - config.train_days
        folds.append(
            Fold(
                dates[train_start],
                dates[train_stop - 1],
                dates[start],
                dates[min(start + config.test_days, len(dates)) - 1],
            )
        )

    return folds


def select_variant(train: Mapping[str, pd.DataFrame]) -> str:
    ranks = {}

    for name in sorted(train):
        if "gross_exposure" not in train[name]:
            raise ValueError(f"{name}: gross_exposure is required for return-on-gross selection")

        pnl = train[name][["net_pnl", "gross_pnl", "gross_exposure"]].dropna()

        if pnl.empty or not np.isfinite(pnl.to_numpy(dtype=float)).all():
            continue

        gross = float(pnl["gross_exposure"].mean())

        if gross <= 0 or (pnl["gross_exposure"] < 0).any():
            continue

        vol = float(pnl["gross_pnl"].std())
        gross_sr = float(pnl["gross_pnl"].mean() / vol) if vol > 0 else -np.inf
        ranks[name] = (float(pnl["net_pnl"].mean()) / gross, gross_sr)

    if not ranks:
        raise ValueError("no variants with usable training P&L")

    return max(ranks, key=lambda name: ranks[name])


SelectFn = Callable[[Mapping[str, pd.DataFrame]], str]


@dataclass(frozen=True)
class WalkForwardResult:
    folds: pd.DataFrame  # [fold] train/test dates, chosen variant, test days
    oos: pd.DataFrame


def run_walk_forward(
    daily: Mapping[str, pd.DataFrame],
    folds: Sequence[Fold],
    select: SelectFn = select_variant,
) -> WalkForwardResult:
    if not daily:
        raise ValueError("no variants")

    for fold in folds:
        if not fold.train_start <= fold.train_end < fold.test_start <= fold.test_end:
            raise ValueError(f"fold dates out of order: {fold}")

    for before, after in itertools.pairwise(folds):
        if after.test_start <= before.test_end:
            raise ValueError(f"test windows overlap: {before} / {after}")

    rows, parts = [], []

    for k, fold in enumerate(folds):
        train = {name: d.loc[fold.train_start : fold.train_end] for name, d in daily.items()}
        choice = select(train)

        if choice not in daily:
            raise KeyError(f"select returned unknown variant {choice!r}")

        test = daily[choice].loc[fold.test_start : fold.test_end].copy()
        test["variant"], test["fold"] = choice, k
        parts.append(test)
        rows.append(
            {
                "fold": k,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "chosen": choice,
                "test_days": len(test),
            }
        )

    folds_df = pd.DataFrame(rows).set_index("fold") if rows else pd.DataFrame()
    oos = pd.concat(parts) if parts else pd.DataFrame()

    return WalkForwardResult(folds=folds_df, oos=oos)


def _annual(pnl: pd.Series) -> dict[str, float]:
    ann_return, ann_vol = float(pnl.mean() * 252), float(pnl.std() * np.sqrt(252))
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else float("nan"),
    }


def oos_metrics(result: WalkForwardResult) -> dict[str, float]:
    out = _annual(result.oos["net_pnl"])
    out |= {f"gross_{k}": v for k, v in _annual(result.oos["gross_pnl"]).items()}
    n_switches = int((result.folds["chosen"] != result.folds["chosen"].shift()).sum() - 1)

    return out | {
        "folds": float(len(result.folds)),
        "switches": float(max(n_switches, 0)),
        "days": float(len(result.oos)),
    }
