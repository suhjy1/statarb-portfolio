from __future__ import annotations

import torch
from torch import Tensor

from statarb.config import BacktestConfig, CostConfig


def allocate(
    preferences: Tensor,
    beta: Tensor,
    etf_index: Tensor,
    n_etfs: int,
    *,
    max_weight: float,
    gross_limit: float,
) -> Tensor:
    if preferences.ndim != 1 or beta.shape != preferences.shape or etf_index.shape != beta.shape:
        raise ValueError("preferences, beta and ETF indices must be aligned vectors")

    if max_weight <= 0 or gross_limit <= 0 or n_etfs < 1:
        raise ValueError("positive portfolio bounds and at least one ETF are required")

    if not len(preferences):
        return preferences.new_zeros(n_etfs)

    c = 1 - beta
    q = preferences * max_weight
    q = q - c * (q @ c) / (c @ c).clamp_min(1e-12)
    hedge = q.new_zeros(n_etfs).index_add(0, etf_index, -beta * q)
    book = torch.cat([q, hedge])
    gross_scale = gross_limit / book.abs().sum().clamp_min(1e-12)
    cap_scale = max_weight / q.abs().max().clamp_min(1e-12)
    scale = torch.minimum(torch.minimum(gross_scale, cap_scale), q.new_tensor(1.0))

    return book * scale


def execution_pnl(
    targets: Tensor,
    returns: Tensor,
    dollar_volume: Tensor,
    is_etf: Tensor,
    config: BacktestConfig,
    costs: CostConfig,
) -> tuple[Tensor, Tensor]:
    if config.min_trade != 0:
        raise ValueError("differentiable training requires backtest.min_trade=0")

    if targets.ndim != 2 or returns.ndim != 2 or targets.shape[1] != returns.shape[1]:
        raise ValueError("targets and returns must be aligned matrices")

    if dollar_volume.shape != returns.shape or is_etf.shape != returns.shape[1:]:
        raise ValueError("volume and ETF mask must align with returns")

    targets = targets.to(dtype=torch.float64)
    returns = torch.where(torch.isfinite(returns), returns, 0).to(dtype=torch.float64)
    dv = dollar_volume.to(dtype=torch.float64)
    w = targets.new_zeros(returns.shape[1])
    net, gross = [], []

    for d in range(len(returns)):
        pnl = w @ returns[d]
        fee = (-w).clamp_min(0).sum() * costs.borrow_bps_annual * 1e-4 / 252
        growth = 1 + pnl
        drift = w * (1 + returns[d])
        w = drift / torch.where(growth > 0, growth, growth.new_tensor(1.0))
        k = d - config.execution_lag + 1

        if 0 <= k < len(targets):
            trade = (targets[k] - w).abs()
            valid_dv = torch.isfinite(dv[d]) & (dv[d] > 0)
            safe_dv = torch.where(valid_dv, dv[d], 1)
            spread = trade.new_full(trade.shape, costs.stock_half_spread_bps)

            if costs.spread_model == "liquidity":
                spread = (
                    costs.stock_half_spread_bps
                    * (costs.spread_ref_dollar_volume / safe_dv) ** costs.spread_elasticity
                )
                spread = torch.where(valid_dv, spread, costs.spread_max_bps).clamp(
                    costs.spread_min_bps, costs.spread_max_bps
                )

            spread = torch.where(is_etf, costs.etf_half_spread_bps, spread)
            fee = fee + (trade * spread * 1e-4).sum()

            if costs.impact_coef > 0:
                impact = torch.where(
                    valid_dv, costs.impact_coef * (costs.capital / safe_dv).sqrt(), 0
                )
                fee = fee + (impact * trade.pow(1.5)).sum()

            w = targets[k]

        net.append(pnl - fee)
        gross.append(pnl)

    if not net:
        return targets.new_empty(0), targets.new_empty(0)

    return torch.stack(net), torch.stack(gross)
