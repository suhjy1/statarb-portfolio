from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from statarb.config import ResearchConfig
from statarb.neural.config import ModernConfig
from statarb.neural.dataset import ModernTape, features_at
from statarb.neural.networks import make_policy
from statarb.neural.portfolio import allocate, execution_pnl
from statarb.panel import ResearchPanel


@dataclass(frozen=True)
class Decision:
    positions: np.ndarray
    features: np.ndarray
    beta: np.ndarray
    etf_index: np.ndarray


def decisions(tape: ModernTape, lookback: int) -> list[Decision]:
    result = []

    for date in tape.dates:
        ids, x = features_at(tape, date, lookback=lookback)
        result.append(
            Decision(
                tape.ids.get_indexer(ids),
                x,
                tape.hedge_beta.loc[date, ids].to_numpy(dtype=float, copy=True),
                tape.etfs.get_indexer(tape.hedge_etf.loc[date, ids]),
            )
        )

    return result


def policy_target(
    model: nn.Module,
    decision: Decision,
    n_stocks: int,
    n_etfs: int,
    research: ResearchConfig,
) -> Tensor:
    if not len(decision.positions):
        return torch.zeros(n_stocks + n_etfs, dtype=torch.float32)

    scores = model(torch.as_tensor(decision.features, dtype=torch.float32))
    book = allocate(
        scores,
        torch.as_tensor(decision.beta, dtype=scores.dtype),
        torch.as_tensor(decision.etf_index, dtype=torch.long),
        n_etfs,
        max_weight=research.portfolio.max_weight,
        gross_limit=research.portfolio.gross_leverage,
    )
    stocks = book.new_zeros(n_stocks).index_add(
        0, torch.as_tensor(decision.positions, dtype=torch.long), book[: len(scores)]
    )

    return torch.cat([stocks, book[len(scores) :]])


def fit_policy(
    tape: ModernTape,
    panel: ResearchPanel,
    research: ResearchConfig,
    config: ModernConfig,
    cache: list[Decision],
    cutoff: pd.Timestamp,
) -> tuple[nn.Module, pd.DataFrame, dict[str, object]]:
    torch.set_num_threads(config.cpu_threads)
    model = make_policy(config.architecture, config.lookback, 2, config.hidden_size, config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dates = tape.dates
    end = int(dates.searchsorted(cutoff, side="right")) - 1
    first = max(
        int(dates.searchsorted(pd.Timestamp(config.train_start))),
        end - config.training_window_days + 1,
    )
    lag, size = research.backtest.execution_lag, config.chunk_days
    starts = [
        k
        for k in range(first, end - lag - size + 2)
        if any(len(cache[j].positions) for j in range(k, k + size))
    ]

    if not starts:
        raise ValueError("not enough matured training history for a complete chunk")

    cols = tape.ids.append(tape.etfs)
    returns = pd.concat([panel.returns, panel.etf_returns], axis=1).reindex(columns=cols)
    dv = panel.dollar_volume.reindex(columns=cols)
    etf_mask = torch.as_tensor(cols.isin(tape.etfs))
    rng = np.random.default_rng(config.seed)
    rows = []
    model.train()
    latest_feature, latest_outcome = first, first

    for step in range(config.training_steps):
        start = int(rng.choice(starts))
        stop = start + size + lag
        assert dates[stop - 1] <= cutoff
        targets = torch.stack(
            [
                policy_target(model, cache[j], len(tape.ids), len(tape.etfs), research)
                for j in range(start, start + size)
            ]
        )
        net, gross = execution_pnl(
            targets,
            torch.as_tensor(returns.iloc[start:stop].to_numpy(dtype=float, copy=True)),
            torch.as_tensor(dv.iloc[start:stop].to_numpy(dtype=float, copy=True)),
            etf_mask,
            research.backtest,
            research.costs,
        )
        loss = -net.mean() + config.risk_aversion * gross.var(unbiased=False)

        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        latest_feature = max(latest_feature, start + size - 1)
        latest_outcome = max(latest_outcome, stop - 1)
        rows.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "mean_net_pnl": float(net.mean().detach()),
                "outcome_end": dates[stop - 1],
            }
        )

    model.eval()
    audit = {
        "cutoff": cutoff,
        "first_training_date": dates[first],
        "latest_feature_date": dates[latest_feature],
        "latest_outcome_date": dates[latest_outcome],
        "eligible_chunks": len(starts),
        "training_steps": config.training_steps,
    }

    return model, pd.DataFrame(rows), audit
