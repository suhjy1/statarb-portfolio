from __future__ import annotations

import portfolio_contracts as contracts
from statarb.portfolio.construction import hedge_weights, stock_weights, target_weights
from statarb.portfolio.costs import borrow_cost, stock_half_spread_bps, trading_cost
from statarb.portfolio.optimizer import (
    expected_alpha,
    optimize_weights,
    optimized_target_weights,
    residual_covariance,
)
from statarb.portfolio.turnover import smooth_target


def test_stock_weights() -> None:
    contracts.check_stock_weights(stock_weights)


def test_hedge_weights() -> None:
    contracts.check_hedge_weights(hedge_weights)


def test_target_weights() -> None:
    contracts.check_target_weights(target_weights)


def test_trading_cost() -> None:
    contracts.check_trading_cost(trading_cost)


def test_borrow_cost() -> None:
    contracts.check_borrow_cost(borrow_cost)


def test_stock_half_spread() -> None:
    contracts.check_stock_half_spread(stock_half_spread_bps)


def test_smooth_target() -> None:
    contracts.check_smooth_target(smooth_target)


def test_expected_alpha() -> None:
    contracts.check_expected_alpha(expected_alpha)


def test_residual_covariance() -> None:
    contracts.check_residual_covariance(residual_covariance)


def test_optimize_weights() -> None:
    contracts.check_optimize_weights(optimize_weights)


def test_optimized_target_weights() -> None:
    contracts.check_optimized_target_weights(optimized_target_weights)
