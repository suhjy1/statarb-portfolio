"""End-to-end walkthrough on a synthetic market with planted mean-reverting residuals

since I plant the ground truth in the generator, the demo first checks that the residual
model and OU fit recover it, and then compares portfolio construction variants

run with python examples/demo.py for the classical pipeline (~1 min)
add --neural to also train a small conv-Transformer policy (needs torch)
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from statarb.backtest.engine import BacktestResult, summarize
from statarb.backtest.runner import run_strategy
from statarb.config import ResearchConfig
from statarb.evaluation.decay import residual_decay_profile
from statarb.evaluation.stats import probabilistic_sharpe_ratio, sharpe_tstat
from statarb.models.pipeline import run_signals
from statarb.synthetic import SECTOR_ETFS, synthetic_market

BASE: dict[str, Any] = {
    "rebalance_every": 5,
    "model": {"kind": "etf", "estimation_window": 252, "etfs": ["SPY", *SECTOR_ETFS]},
    "ou": {"window": 60},
    "backtest": {"execution_lag": 2, "min_trade": 0.002},
}
VARIANTS: dict[str, dict[str, Any]] = {
    "rule (baseline)": {},
    "rule + smoothing h=4": {"turnover": {"smoothing_halflife": 4.0}, "backtest": {"min_trade": 0}},
    "calibrated optimizer H=3": {
        "portfolio": {"construction": "optimizer"},
        "optimizer": {"horizon": 3, "alpha_calibration": "online", "calib_min_rebalances": 8},
    },
}


def research_config(**overrides: Any) -> ResearchConfig:
    merged = {k: BASE.get(k, {}) | overrides.get(k, {}) for k in ("model", "ou", "backtest")}
    rest = {k: v for k, v in overrides.items() if k not in merged}
    return ResearchConfig.model_validate(BASE | rest | merged)


def report(bt: BacktestResult) -> dict[str, float]:
    m = summarize(bt)
    return {
        "net %/yr": 100 * m["ann_return"],
        "sharpe": m["sharpe"],
        "HAC t": sharpe_tstat(bt.net_pnl),
        "PSR": probabilistic_sharpe_ratio(bt.net_pnl),
        "cost %/yr": 100 * m["cost_drag"],
        "turnover/day": m["avg_turnover"],
        "gross": m["avg_gross"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--neural", action="store_true", help="also train a conv-Transformer policy")
    args = ap.parse_args()

    base = research_config()

    # 1. identification, check the planted structure is found on pure OU and random walk residuals
    clean = synthetic_market(n_stocks=150, n_days=900, noise_vol=0.0, seed=0)
    truth = clean.truth
    diag = run_signals(clean.panel, base).diagnostics
    kappa = diag.groupby("security_id")["kappa"].median()
    passed = diag.groupby("security_id").size().div(diag["date"].nunique())
    rev, walk = truth.index[truth.reverting], truth.index[~truth.reverting]
    print(
        f"1. identification on clean residuals ({len(truth)} stocks, "
        f"{truth.reverting.mean():.0%} OU, rest random walk)"
    )
    print(
        f"   median fitted kappa/yr: reverting {kappa.reindex(rev).median():.0f} "
        f"(true {truth.kappa[rev].median():.0f}), random walk {kappa.reindex(walk).median():.0f} "
        "(true 0: Dickey-Fuller small-sample bias)"
    )
    print(
        f"   rank corr(fitted, true kappa) on reverting names: "
        f"{kappa.reindex(rev).corr(truth.kappa[rev], method='spearman'):.2f}"
    )
    print(
        f"   rebalances passing kappa >= {base.ou.min_kappa:.1f}: reverting "
        f"{passed.reindex(rev).fillna(0).mean():.0%}, random walk "
        f"{passed.reindex(walk).fillna(0).mean():.0%}\n"
    )

    # 2-4. trading, on residuals diluted by non-reverting idiosyncratic noise
    panel = synthetic_market(n_stocks=150, n_days=900, seed=0).panel
    rows = {}
    for name, overrides in VARIANTS.items():
        rows[name] = report(run_strategy(panel, research_config(**overrides)).backtest)
    print("2. portfolio construction variants (net of 5 bp spreads + 50 bp borrow)")
    print(pd.DataFrame(rows).T.round(3).to_string(), "\n")

    # 3. signal decay
    decay = residual_decay_profile(panel, base, horizons=[1, 5, 10, 20])
    print("3. s-score rank IC vs forward residual return (negative = mean reversion)")
    print(decay.round(4).to_string(), "\n")

    if args.neural:
        from statarb.neural.config import ModernConfig
        from statarb.neural.runner import run_modern

        dates = panel.dates
        cfg = ModernConfig(
            architecture="conv_transformer",
            train_start=dates[0].date(),
            train_end=dates[499].date(),
            test_start=dates[500].date(),
            test_end=dates[-1].date(),
            training_window_days=400,
            refit_every=200,
            training_steps=150,
            max_stocks=60,
        )
        res = run_modern(panel, base, cfg)
        print("4. conv-Transformer policy, out-of-sample")
        print(pd.Series(report(res.backtest)).round(3).to_string())


if __name__ == "__main__":
    main()
