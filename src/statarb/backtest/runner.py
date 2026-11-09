from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from statarb.backtest.engine import BacktestResult, BorrowCostFn, TradingCostFn, run_backtest
from statarb.config import PortfolioConfig, ResearchConfig, TurnoverConfig
from statarb.models.pipeline import OnRebalance, RebalanceContext, SignalResult, run_signals
from statarb.panel import ResearchPanel
from statarb.portfolio.construction import target_weights
from statarb.portfolio.costs import borrow_cost, trading_cost
from statarb.portfolio.optimizer import AlphaCalibrator, expected_alpha, optimized_target_weights
from statarb.portfolio.turnover import smooth_target

log = logging.getLogger(__name__)

WeightsFn = Callable[[RebalanceContext, PortfolioConfig], pd.Series]
OptimizerFn = Callable[[RebalanceContext, pd.Series, ResearchConfig], pd.Series]
SmoothingFn = Callable[[pd.Series, pd.Series, TurnoverConfig], pd.Series]
BacktestFn = Callable[..., BacktestResult]


@dataclass(frozen=True)
class StrategyResult:
    signals: SignalResult
    targets: pd.DataFrame  # [rebalance date x instrument]
    backtest: BacktestResult
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)


def instrument_returns(panel: ResearchPanel) -> pd.DataFrame:
    return pd.concat([panel.returns, panel.etf_returns], axis=1)


def rebalance_step(
    config: ResearchConfig,
    *,
    weights_fn: WeightsFn | None = None,
    optimizer_fn: OptimizerFn = optimized_target_weights,
    smoothing_fn: SmoothingFn = smooth_target,
    calibration_rows: list[dict[str, object]] | None = None,
) -> OnRebalance:
    previous = pd.Series(dtype=float)
    calibrator = (
        AlphaCalibrator(config)
        if config.portfolio.construction == "optimizer"
        and config.optimizer.alpha_calibration == "online"
        else None
    )

    if calibrator is not None and (
        weights_fn is not None or optimizer_fn is not optimized_target_weights
    ):
        raise ValueError("online calibration requires the default optimizer")

    def step(ctx: RebalanceContext) -> pd.Series:
        nonlocal previous

        if weights_fn is not None:
            target = weights_fn(ctx, config.portfolio)
        elif config.portfolio.construction == "optimizer":
            if calibrator is None:
                target = optimizer_fn(ctx, previous, config)
            else:
                alpha = calibrator.calibrate(ctx, expected_alpha(ctx, config))
                target = optimized_target_weights(ctx, previous, config, alpha_override=alpha)

                if calibration_rows is not None:
                    calibration_rows.append(calibrator.diagnostics[-1])
        else:
            target = target_weights(ctx, config.portfolio)

        if config.turnover.smoothing_halflife > 0:
            target = smoothing_fn(target, previous, config.turnover)

        previous = target

        return target

    return step


def run_strategy(
    panel: ResearchPanel,
    config: ResearchConfig,
    *,
    weights_fn: WeightsFn | None = None,
    optimizer_fn: OptimizerFn = optimized_target_weights,
    smoothing_fn: SmoothingFn = smooth_target,
    trading_cost_fn: TradingCostFn = trading_cost,
    borrow_cost_fn: BorrowCostFn = borrow_cost,
    backtest_fn: BacktestFn = run_backtest,
    **signal_components: Any,
) -> StrategyResult:
    calibration_rows: list[dict[str, object]] = []
    step = rebalance_step(
        config,
        weights_fn=weights_fn,
        optimizer_fn=optimizer_fn,
        smoothing_fn=smoothing_fn,
        calibration_rows=calibration_rows,
    )
    signals = run_signals(panel, config, on_rebalance=step, **signal_components)
    targets = signals.targets if signals.targets is not None else pd.DataFrame()
    days = panel.signal_dates
    returns = instrument_returns(panel).loc[days]
    result = backtest_fn(
        targets,
        returns,
        panel.dollar_volume.loc[days],
        config.backtest,
        config.costs,
        etfs=set(panel.etf_returns.columns),
        trading_cost_fn=trading_cost_fn,
        borrow_cost_fn=borrow_cost_fn,
    )
    log.info("strategy: %d rebalances, %d backtest days", len(targets), len(result.net_pnl))

    return StrategyResult(
        signals=signals,
        targets=targets,
        backtest=result,
        calibration=pd.DataFrame(calibration_rows),
    )
