from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from statarb.backtest.engine import run_backtest
from statarb.config import BacktestConfig, CostConfig
from statarb.neural.portfolio import allocate, execution_pnl


@pytest.mark.parametrize("unit_beta", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gross_limit", [0.035, 2.0])
def test_allocation_is_tradable_factor_hedged_neutral_and_bounded(
    unit_beta: bool, dtype: torch.dtype, gross_limit: float
) -> None:
    preference = torch.tensor([-0.4, 1.5, 0.1, -1.1, 0.8, -0.7], dtype=dtype, requires_grad=True)
    beta = (
        torch.ones(6, dtype=dtype)
        if unit_beta
        else torch.tensor([0.8, 1.2, 1.0, 0.7, 1.6, 1.1], dtype=dtype)
    )
    assigned = torch.tensor([0, 0, 1, 1, 2, 0], dtype=torch.int64)
    book = allocate(preference, beta, assigned, 3, max_weight=0.02, gross_limit=gross_limit)
    tolerance = 1e-7 if dtype == torch.float32 else 1e-12
    assert book.shape == (9,)
    assert torch.isfinite(book).all()
    assert abs(float(book.sum().detach())) < tolerance
    assert float(book[:6].abs().max().detach()) <= 0.02 + tolerance
    assert float(book.abs().sum().detach()) <= gross_limit + tolerance

    factors = torch.tensor([0.012, -0.02, 0.03], dtype=dtype)
    residuals = torch.tensor([0.01, -0.005, 0.02, -0.012, 0.004, 0.007], dtype=dtype)
    stock_returns = beta * factors[assigned] + residuals
    actual = book @ torch.cat([stock_returns, factors])
    expected = book[:6] @ residuals
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
    actual.backward()
    assert preference.grad is not None and torch.isfinite(preference.grad).all()


def test_allocation_respects_permutation_and_small_preferences_remain_small() -> None:
    preference = torch.tensor([0.1, -0.2, 0.15, -0.1], dtype=torch.float64)
    beta = torch.tensor([0.8, 1.2, 0.7, 1.1], dtype=torch.float64)
    assigned = torch.tensor([0, 1, 1, 0])
    kwargs = {"n_etfs": 2, "max_weight": 0.02, "gross_limit": 2.0}
    book = allocate(preference, beta, assigned, **kwargs)
    small = allocate(preference * 0.01, beta, assigned, **kwargs)
    torch.testing.assert_close(small, book * 0.01, atol=1e-15, rtol=0)
    perm = torch.tensor([2, 0, 3, 1])
    permuted = allocate(preference[perm], beta[perm], assigned[perm], **kwargs)
    torch.testing.assert_close(permuted[:4][torch.argsort(perm)], book[:4], atol=1e-15, rtol=0)
    torch.testing.assert_close(permuted[4:], book[4:], atol=1e-15, rtol=0)


def test_empty_and_infeasible_single_stock_books() -> None:
    empty = allocate(
        torch.empty(0),
        torch.empty(0),
        torch.empty(0, dtype=torch.int64),
        2,
        max_weight=0.02,
        gross_limit=2,
    )
    assert torch.equal(empty, torch.zeros(2))
    one = allocate(
        torch.tensor([0.9], dtype=torch.float64),
        torch.tensor([0.8], dtype=torch.float64),
        torch.tensor([0]),
        1,
        max_weight=0.02,
        gross_limit=2,
    )
    torch.testing.assert_close(one, torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)


def world() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    rng = np.random.default_rng(30)
    dates = pd.bdate_range("2022-01-03", periods=12)
    stocks = [1001, 1002, 1003, 1004]
    etfs = {"SPY", "XLK"}
    instruments = [*stocks, "SPY", "XLK"]
    targets = pd.DataFrame(rng.normal(0, 0.02, (8, 6)), index=dates[:8], columns=instruments)
    targets.iloc[3, 1:4] = 0.0
    targets.iloc[4] = targets.iloc[3]  # same target may still trade after weight drift
    returns = pd.DataFrame(rng.normal(0, 0.015, (12, 6)), index=dates, columns=instruments)
    returns.iloc[2, 0] = np.nan
    returns.iloc[4, 2] = np.inf
    volume = pd.DataFrame(rng.uniform(1e6, 1e9, (12, 6)), index=dates, columns=instruments)
    volume.loc[:, ["SPY", "XLK"]] = np.nan
    volume.iloc[0, 0] = np.nan
    volume.iloc[1, 1] = 0.0
    volume.iloc[2, 2] = -1.0
    volume.iloc[3, 3] = np.inf

    return targets, returns, volume, etfs


def assert_execution_matches(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    etfs: set[str],
    config: BacktestConfig,
    costs: CostConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = run_backtest(targets, returns, volume, config, costs, etfs=etfs)
    net, gross = execution_pnl(
        torch.tensor(targets.to_numpy(), dtype=torch.float64),
        torch.tensor(returns.to_numpy(), dtype=torch.float64),
        torch.tensor(volume.to_numpy(), dtype=torch.float64),
        torch.tensor(returns.columns.isin(etfs), dtype=torch.bool),
        config,
        costs,
    )
    np.testing.assert_allclose(net.detach().numpy(), expected.net_pnl, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gross.detach().numpy(), expected.gross_pnl, atol=1e-12, rtol=1e-12)

    return net, gross


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("spread_model", ["constant", "liquidity"])
@pytest.mark.parametrize("impact", [0.0, 0.07])
def test_execution_matches_reference_with_drift_costs_borrow_and_missing_data(
    lag: int, spread_model: str, impact: float
) -> None:
    assert_execution_matches(
        *world(),
        BacktestConfig(execution_lag=lag),
        CostConfig(spread_model=spread_model, impact_coef=impact),
    )


@pytest.mark.parametrize("zero_book", [False, True])
def test_zero_changes_have_no_repeated_trading_fee_and_finite_gradients(zero_book: bool) -> None:
    dates = pd.bdate_range("2022-01-03", periods=5)
    weights = [0, 0, 0] if zero_book else [0.2, -0.1, -0.1]
    targets = pd.DataFrame([weights] * 5, index=dates, columns=[1001, 1002, "SPY"])
    returns = pd.DataFrame(0.0, index=dates, columns=targets.columns)
    dv = pd.DataFrame(1e8, index=dates, columns=targets.columns)
    dv["SPY"] = np.nan
    costs = CostConfig(impact_coef=0.1, borrow_bps_annual=252)
    net, _ = assert_execution_matches(targets, returns, dv, {"SPY"}, BacktestConfig(), costs)

    if zero_book:
        assert torch.equal(net, torch.zeros(5, dtype=torch.float64))
    else:
        np.testing.assert_allclose(net.numpy()[1:], -0.2 * 1e-4, atol=1e-14)

    differentiable = torch.tensor(targets.to_numpy(), dtype=torch.float64, requires_grad=True)
    loss, _ = execution_pnl(
        differentiable,
        torch.tensor(returns.to_numpy()),
        torch.tensor(dv.to_numpy()),
        torch.tensor([False, False, True]),
        BacktestConfig(),
        costs,
    )
    loss.sum().backward()
    assert differentiable.grad is not None and torch.isfinite(differentiable.grad).all()


@pytest.mark.parametrize("return_on_loss_day", [-1.0, -1 + 5e-14, -1.2])
def test_drift_growth_branch_matches_reference_at_and_around_total_capital_loss(
    return_on_loss_day: float,
) -> None:
    dates = pd.bdate_range("2022-01-03", periods=4)
    targets = pd.DataFrame({1001: [1.0]}, index=dates[:1])
    returns = pd.DataFrame({1001: [0, return_on_loss_day, 0.2, 0.1]}, index=dates)
    volume = pd.DataFrame(1e8, index=dates, columns=[1001])
    assert_execution_matches(
        targets,
        returns,
        volume,
        set(),
        BacktestConfig(),
        CostConfig(stock_half_spread_bps=0, borrow_bps_annual=0),
    )


def test_training_refuses_a_discontinuous_no_trade_band() -> None:
    targets, returns, volume, etfs = world()
    with pytest.raises(ValueError, match="min_trade"):
        execution_pnl(
            torch.tensor(targets.to_numpy()),
            torch.tensor(returns.to_numpy()),
            torch.tensor(volume.to_numpy()),
            torch.tensor(returns.columns.isin(etfs)),
            BacktestConfig(min_trade=0.002),
            CostConfig(),
        )
