from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from statarb.evaluation.walk_forward import Fold, run_walk_forward, select_variant

DATES = pd.bdate_range("2015-01-02", periods=400)


def daily_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1)
    out = {}

    for name in ("a", "b", "c"):
        gross = rng.normal(0.0002, 0.01, len(DATES))
        out[name] = pd.DataFrame(
            {
                "gross_pnl": gross,
                "costs": 1e-4,
                "net_pnl": gross - 1e-4,
                "turnover": 0.3,
                "gross_exposure": 1.0,
            },
            index=DATES,
        )

    return out


def simple_folds(train: int = 100, test: int = 50, embargo: int = 2) -> list[Fold]:
    # expanding train, contiguous test windows
    folds, start = [], train + embargo

    while start < len(DATES):
        end = min(start + test, len(DATES)) - 1
        folds.append(Fold(DATES[0], DATES[start - embargo - 1], DATES[start], DATES[end]))
        start = end + 1

    return folds


def simple_select(train: Mapping[str, pd.DataFrame]) -> str:
    # highest mean net P&L, ties broken by name
    return max(sorted(train), key=lambda k: float(train[k]["net_pnl"].mean()))


def test_selection_sees_only_train_rows_and_oos_is_stitched() -> None:
    daily, folds = daily_tables(), simple_folds()
    seen: list[pd.Timestamp] = []

    def recording(train: Mapping[str, pd.DataFrame]) -> str:
        seen.append(max(d.index.max() for d in train.values()))
        return simple_select(train)

    res = run_walk_forward(daily, folds, recording)
    assert seen == [f.train_end for f in folds]
    assert res.oos.index.is_unique and res.oos.index.is_monotonic_increasing
    assert res.oos.index[0] == folds[0].test_start and res.oos.index[-1] == DATES[-1]
    in_tests = sum(len(DATES[DATES.slice_indexer(f.test_start, f.test_end)]) for f in folds)
    assert len(res.oos) == in_tests

    for k, fold in res.folds.iterrows():
        chunk = res.oos[res.oos["fold"] == k]
        pd.testing.assert_series_equal(
            chunk["net_pnl"], daily[fold["chosen"]].loc[chunk.index, "net_pnl"]
        )


@pytest.mark.parametrize("real_selection", [False, True])
def test_walk_forward_has_no_lookahead(real_selection: bool) -> None:
    daily, folds = daily_tables(), simple_folds()
    select = select_variant if real_selection else simple_select
    full = run_walk_forward(daily, folds, select)
    k = 3
    cut = folds[k].test_end
    later = {n: d.copy() for n, d in daily.items()}

    for d in later.values():
        d.loc[d.index > cut, ["gross_pnl", "net_pnl"]] = 1.0  # wildly different future

    trunc = run_walk_forward(later, folds, select)
    pd.testing.assert_frame_equal(full.folds.loc[:k], trunc.folds.loc[:k])
    pd.testing.assert_frame_equal(full.oos.loc[:cut], trunc.oos.loc[:cut])


def test_overlapping_or_unordered_folds_are_refused() -> None:
    daily, folds = daily_tables(), simple_folds()
    overlap = [folds[0], Fold(DATES[0], DATES[100], folds[0].test_end, DATES[200])]

    with pytest.raises(ValueError, match="overlap"):
        run_walk_forward(daily, overlap, simple_select)

    leaky = [Fold(DATES[0], DATES[150], DATES[150], DATES[200])]

    with pytest.raises(ValueError, match="out of order"):
        run_walk_forward(daily, leaky, simple_select)

    with pytest.raises(KeyError):
        run_walk_forward(daily, folds, lambda _: "nope")
