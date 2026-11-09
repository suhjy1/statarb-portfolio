from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from statarb.backtest.engine import run_backtest
from statarb.config import BacktestConfig, CostConfig
from statarb.models.ou import fit_ou, simulate_ou

REPO_ROOT = Path(__file__).resolve().parents[3]

OUT = REPO_ROOT / "reports/benchmarks/bench.csv"


def best_time(fn: Callable[[], Any], repeat: int) -> float:
    times = []

    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    return min(times)


def _cpp_time(make: Callable[[], Callable[[], Any]], repeat: int) -> tuple[float, str]:
    try:
        fn = make()
        fn()  # warm up, also surfaces NotImplementedError
    except ImportError:
        return float("nan"), "not built"
    except NotImplementedError:
        return float("nan"), "not implemented"
    return best_time(fn, repeat), ""


def bench_ou(paths: int, window: int, repeat: int) -> dict[str, Any]:
    frame = pd.DataFrame({i: simulate_ou(15.0, 0.0, 0.2, window, seed=i) for i in range(paths)})

    def python() -> None:
        for sid in frame.columns:
            fit_ou(frame[sid].to_numpy())

    def make_cpp() -> Callable[[], Any]:
        from statarb.native.ou import fit_ou_columns

        return lambda: fit_ou_columns(frame)

    cpp, note = _cpp_time(make_cpp, repeat)

    return {
        "function": "ou_fit",
        "size": f"{paths} paths x {window} days",
        "python_s": best_time(python, repeat),
        "cpp_s": cpp,
        "note": note,
    }


def bench_backtest(days: int, instruments: int, repeat: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2010-01-04", periods=days)
    cols: list[object] = [*range(instruments - 2), "SPY", "XLK"]
    r = pd.DataFrame(rng.normal(0.0, 0.02, (days, len(cols))), index=dates, columns=cols)
    when = dates[::5]
    t = pd.DataFrame(rng.normal(0.0, 0.004, (len(when), len(cols))), index=when, columns=cols)
    dv = pd.DataFrame(1e8, index=dates, columns=cols[:-2])
    kwargs = {
        "targets": t,
        "returns": r,
        "dollar_volume": dv,
        "config": BacktestConfig(execution_lag=2, min_trade=0.002),
        "costs": CostConfig(impact_coef=0.1),
        "etfs": {"SPY", "XLK"},
    }

    def make_cpp() -> Callable[[], Any]:
        from statarb.native.backtest import run_backtest_cpp

        return lambda: run_backtest_cpp(**kwargs)  # type: ignore[arg-type]

    cpp, note = _cpp_time(make_cpp, repeat)
    python = best_time(lambda: run_backtest(**kwargs), repeat)  # type: ignore[arg-type]

    return {
        "function": "backtest",
        "size": f"{days} days x {instruments} instruments",
        "python_s": python,
        "cpp_s": cpp,
        "note": note,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Python vs C++ timings.")
    ap.add_argument("--paths", type=int, default=1500)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--days", type=int, default=1000)
    ap.add_argument("--instruments", type=int, default=800)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    table = pd.DataFrame(
        [
            bench_ou(args.paths, args.window, args.repeat),
            bench_backtest(args.days, args.instruments, args.repeat),
        ]
    )
    table["speedup"] = table["python_s"] / table["cpp_s"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
