from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from statarb.backtest.engine import BacktestResult, run_backtest
from statarb.config import ResearchConfig
from statarb.neural.config import ModernConfig
from statarb.neural.dataset import build_tape
from statarb.neural.training import decisions, fit_policy, policy_target
from statarb.panel import ResearchPanel


@dataclass(frozen=True)
class ModernResult:
    targets: pd.DataFrame
    backtest: BacktestResult
    fit_audit: pd.DataFrame
    training: pd.DataFrame


def extension_research(research: ResearchConfig, config: ModernConfig) -> ResearchConfig:
    costs = research.costs

    if config.borrow_bps_annual is not None:
        costs = costs.model_copy(update={"borrow_bps_annual": config.borrow_bps_annual})

    return research.model_copy(
        update={
            "backtest": research.backtest.model_copy(update={"min_trade": 0.0}),
            "costs": costs,
            "portfolio": research.portfolio.model_copy(
                update={"hedge": True, "dollar_neutral": True}
            ),
        }
    )


def run_modern(
    panel: ResearchPanel,
    research: ResearchConfig,
    config: ModernConfig,
    *,
    checkpoint_dir: Path | None = None,
) -> ModernResult:
    research = extension_research(research, config)
    tape = build_tape(panel, research, max_stocks=config.max_stocks)
    cache = decisions(tape, config.lookback)
    test = tape.dates[
        (tape.dates >= pd.Timestamp(config.test_start))
        & (tape.dates <= pd.Timestamp(config.test_end))
    ]

    if test.empty:
        raise ValueError("no test trading dates")

    cols = tape.ids.append(tape.etfs)
    targets, audits, training = {}, [], []
    model = None

    for k, date in enumerate(test):
        pos = int(tape.dates.get_loc(date))

        if pos == 0:
            raise ValueError("test needs prior trading history")

        if k % config.refit_every == 0:
            cutoff = tape.dates[pos - 1]

            if k == 0:
                cutoff = min(cutoff, pd.Timestamp(config.train_end))

            model, losses, audit = fit_policy(tape, panel, research, config, cache, cutoff)
            audit |= {"refit": len(audits), "decision_date": date}
            audits.append(audit)
            losses["refit"] = len(audits) - 1
            training.append(losses)

            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config.model_dump(mode="json"),
                        "audit": {key: str(value) for key, value in audit.items()},
                    },
                    checkpoint_dir / f"fit_{len(audits) - 1:03d}.pt",
                )

        assert model is not None

        with torch.no_grad():
            weights = policy_target(model, cache[pos], len(tape.ids), len(tape.etfs), research)

        targets[date] = weights.detach().numpy().astype(float)

    frame = pd.DataFrame.from_dict(targets, orient="index", columns=cols)
    frame.index.name = "date"
    market = pd.concat([panel.returns, panel.etf_returns], axis=1).reindex(index=test, columns=cols)
    result = run_backtest(
        frame,
        market,
        panel.dollar_volume.reindex(test),
        research.backtest,
        research.costs,
        etfs=set(tape.etfs),
    )

    if not np.isfinite(result.net_pnl).all():
        raise RuntimeError("nonfinite out-of-sample P&L")

    return ModernResult(
        frame,
        result,
        pd.DataFrame(audits),
        pd.concat(training, ignore_index=True),
    )
