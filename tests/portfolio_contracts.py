from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection

import numpy as np
import pandas as pd

from statarb.config import CostConfig, PortfolioConfig, ResearchConfig, TurnoverConfig
from statarb.models.pipeline import RebalanceContext
from statarb.models.residuals import FittedEtf, FittedModel, PcaResidualModel
from statarb.models.signals import FLAT, LONG, SHORT

TOL = 1e-9

StockWeightsFn = Callable[[RebalanceContext, PortfolioConfig], pd.Series]
HedgeWeightsFn = Callable[[pd.Series, FittedModel], pd.Series]
TargetWeightsFn = Callable[[RebalanceContext, PortfolioConfig], pd.Series]
TradingCostFn = Callable[[pd.Series, pd.Series, CostConfig, Collection[str]], pd.Series]
BorrowCostFn = Callable[[pd.Series, CostConfig], pd.Series]
SpreadFn = Callable[[pd.Series, CostConfig], pd.Series]
SmoothTargetFn = Callable[[pd.Series, pd.Series, TurnoverConfig], pd.Series]
AlphaFn = Callable[[RebalanceContext, ResearchConfig], pd.Series]
CovarianceFn = Callable[[RebalanceContext], pd.DataFrame]
OptimizeFn = Callable[[pd.Series, pd.DataFrame, pd.Series, pd.Series, ResearchConfig], pd.Series]
OptimizedTargetFn = Callable[[RebalanceContext, pd.Series, ResearchConfig], pd.Series]


def _held(w: pd.Series) -> pd.Series:
    return w[w.abs() > TOL]


def _etf_fit(ids: list[int]) -> FittedEtf:
    etf = ["XLK" if i % 2 else "XLF" for i in ids]
    beta = [0.8 + 0.1 * (i % 5) for i in ids]
    return FittedEtf(
        etf=pd.Series(etf, index=ids, dtype=object),
        beta=pd.Series(beta, index=ids, dtype=float),
        alpha=pd.Series(0.0, index=ids),
        drop_alpha=True,
    )


def make_context(
    positions: dict[int, int],
    vols: dict[int, float] | None = None,
    fitted: FittedModel | None = None,
    days: int = 120,
) -> RebalanceContext:
    ids = list(positions)
    vols = vols or {}
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=days)
    sd = np.array([vols.get(i, 0.02) for i in ids])
    returns = pd.DataFrame(rng.standard_normal((days, len(ids))) * sd, index=dates, columns=ids)

    return RebalanceContext(
        date=dates[-1],
        positions=pd.Series(positions, dtype="int64"),
        scores=pd.Series({i: -2.0 * p for i, p in positions.items()}, dtype=float),
        fitted=fitted if fitted is not None else _etf_fit(ids),
        returns=returns,
        dollar_volume=pd.Series(5e7, index=ids),
    )


def check_stock_weights(fn: StockWeightsFn) -> None:
    ctx = make_context({1: LONG, 2: LONG, 3: SHORT, 4: FLAT, 5: SHORT})
    cfg = PortfolioConfig(sizing="equal", gross_leverage=2.0, max_weight=1.0)
    w = _held(fn(ctx, cfg))
    assert set(w.index) == {1, 2, 3, 5}, f"held ids {sorted(w.index)} != [1, 2, 3, 5]"
    assert (w[[1, 2]] > 0).all() and (w[[3, 5]] < 0).all(), "LONG must be > 0, SHORT < 0"
    assert np.allclose(w.abs(), 0.5), f"equal sizing, gross 2, 4 names -> |w| = 0.5; got {w}"

    capped = _held(fn(ctx, cfg.model_copy(update={"max_weight": 0.3})))
    assert (capped.abs() <= 0.3 + TOL).all(), f"|w| must respect max_weight 0.3: {capped}"
    assert capped.abs().sum() <= 2.0 + TOL, "sum |w| must not exceed gross_leverage"

    vol_ctx = make_context({1: LONG, 2: LONG, 3: SHORT}, vols={1: 0.02, 2: 0.04, 3: 0.03})
    iv = fn(vol_ctx, cfg.model_copy(update={"sizing": "inverse_vol"}))
    assert abs(iv[1]) > abs(iv[3]) > abs(iv[2]), f"inverse_vol: lower vol -> larger |w|: {iv}"
    assert abs(iv.abs().sum() - 2.0) < 1e-6, "inverse_vol must also hit gross_leverage"

    flat = make_context({1: FLAT, 2: FLAT})
    assert _held(fn(flat, cfg)).empty, "nothing held when every position is FLAT"


def _hedged_pnl_matches_residuals(
    stocks: pd.Series, hedge: pd.Series, fitted: FittedModel, returns: pd.DataFrame
) -> None:
    import dataclasses

    keep_alpha = dataclasses.replace(fitted, drop_alpha=False)  # type: ignore[type-var]
    stock_cols = [c for c in returns.columns if not isinstance(c, str)]
    regressors = returns[[c for c in returns.columns if isinstance(c, str)]]
    resid = keep_alpha.residuals(returns[stock_cols], regressors)
    book = stocks.add(hedge, fill_value=0.0)
    unknown = set(book.index) - set(returns.columns)
    assert not unknown, f"hedge uses instruments with no returns: {sorted(map(str, unknown))}"
    pnl = returns[book.index].to_numpy() @ book.to_numpy()
    expected = resid[stocks.index].to_numpy() @ stocks.to_numpy()
    assert np.allclose(pnl, expected, atol=1e-12), (
        "(stocks + hedge) · r must equal stocks · residual(r) on every day; "
        f"max diff {np.abs(pnl - expected).max():.3g}"
    )


def check_hedge_weights(fn: HedgeWeightsFn) -> None:
    rng = np.random.default_rng(11)
    fitted = FittedEtf(
        etf=pd.Series({1: "XLK", 2: "XLK", 3: "XLF"}, dtype=object),
        beta=pd.Series({1: 1.2, 2: 0.8, 3: 1.0}),
        alpha=pd.Series({1: 0.0, 2: 0.0, 3: 0.0}),
        drop_alpha=True,
    )
    stocks = pd.Series({1: 0.1, 2: -0.05, 3: 0.1})
    h = _held(fn(stocks, fitted))
    expected = pd.Series({"XLK": -(0.1 * 1.2 - 0.05 * 0.8), "XLF": -0.1})
    assert set(h.index) == set(expected.index), f"ETF hedge instruments {list(h.index)}"
    assert np.allclose(h[expected.index], expected), f"ETF hedge {h.to_dict()} != {expected}"
    cols: list[object] = [1, 2, 3, "XLK", "XLF"]
    r = pd.DataFrame(rng.normal(0, 0.01, (30, 5)), columns=cols)
    _hedged_pnl_matches_residuals(stocks, h, fitted, r)

    t, n = 300, 25
    f = rng.normal(0, 0.01, (t + 30, 2))
    loads = rng.uniform(0.5, 1.5, (2, n))
    world = f @ loads + rng.normal(0, 0.005, (t + 30, n))
    fit_rows = pd.DataFrame(world[:t], columns=range(n))
    pca = PcaResidualModel(k=2).fit(fit_rows, pd.DataFrame(index=fit_rows.index))
    stocks = pd.Series({0: 0.1, 3: -0.08, 7: 0.05, 12: -0.07})
    h = fn(stocks, pca)
    new = pd.DataFrame(world[t:], columns=range(n))
    _hedged_pnl_matches_residuals(stocks, h, pca, new)


def check_target_weights(fn: TargetWeightsFn) -> None:
    positions = {1: LONG, 2: LONG, 3: LONG, 4: SHORT, 5: SHORT, 6: FLAT}
    ctx = make_context(positions)
    cfg = PortfolioConfig(gross_leverage=2.0, max_weight=1.0, hedge=True, dollar_neutral=True)
    w = _held(fn(ctx, cfg))
    assert w.abs().sum() <= 2.0 + 1e-6, f"sum |w| = {w.abs().sum():.4f} > gross_leverage 2"
    assert abs(w.sum()) < 1e-6, f"dollar_neutral: sum w = {w.sum():.3g}, expected 0"
    assert any(isinstance(i, str) for i in w.index), "hedge=True must hold the ETF hedges"
    assert 6 not in w.index, "FLAT stock 6 must not be held (except via a hedge basket)"

    unhedged = _held(fn(ctx, cfg.model_copy(update={"hedge": False, "dollar_neutral": False})))
    assert not any(isinstance(i, str) for i in unhedged.index), "hedge=False: no ETF legs"

    flat = make_context({1: FLAT, 2: FLAT})
    assert _held(fn(flat, cfg)).empty, "nothing held when every position is FLAT"


def check_trading_cost(fn: TradingCostFn) -> None:
    cfg = CostConfig(stock_half_spread_bps=5.0, etf_half_spread_bps=1.0, impact_coef=0.0)
    trades = pd.Series({1: 0.1, 2: -0.2, 3: 0.0, "SPY": 0.3})
    dv = pd.Series({1: 1e8, 2: 1e8, 3: 1e8})
    c = fn(trades, dv, cfg, {"SPY"})
    expected = pd.Series({1: 0.1 * 5e-4, 2: 0.2 * 5e-4, 3: 0.0, "SPY": 0.3 * 1e-4})
    assert np.allclose(c.reindex(expected.index).fillna(0.0), expected), (
        f"spread cost {c.to_dict()} != {expected.to_dict()} (|trade| x half-spread)"
    )
    assert (c >= 0).all(), "costs are non-negative"

    imp = cfg.model_copy(update={"impact_coef": 0.1})
    small = fn(pd.Series({1: 0.01}), dv, imp, ())[1]
    large = fn(pd.Series({1: 0.02}), dv, imp, ())[1]
    assert large > 2 * small, "with impact, doubling a trade must more than double its cost"
    illiquid = fn(pd.Series({1: 0.01}), pd.Series({1: 1e6}), imp, ())[1]
    assert illiquid > small, "with impact, the same trade costs more in a less liquid stock"


def check_borrow_cost(fn: BorrowCostFn) -> None:
    cfg = CostConfig(borrow_bps_annual=50.0)
    c = fn(pd.Series({1: 0.1, 2: -0.2, "SPY": -0.1}), cfg)
    daily = 50e-4 / 252
    expected = pd.Series({1: 0.0, 2: 0.2 * daily, "SPY": 0.1 * daily})
    assert np.allclose(c.reindex(expected.index).fillna(0.0), expected), (
        f"borrow {c.to_dict()} != {expected.to_dict()}"
    )


def check_stock_half_spread(fn: SpreadFn) -> None:
    cfg = CostConfig(
        spread_model="liquidity",
        stock_half_spread_bps=5.0,
        spread_ref_dollar_volume=1e8,
        spread_elasticity=0.5,
        spread_min_bps=1.0,
        spread_max_bps=100.0,
    )
    dv = pd.Series({1: 1e5, 2: 1e6, 3: 1e7, 4: 1e8, 5: 1e9, 6: 1e10})
    s = fn(dv, cfg)
    assert s.index.equals(dv.index), f"index {list(s.index)} != {list(dv.index)}"
    assert np.isfinite(s.to_numpy()).all(), f"spreads must be finite: {s.to_dict()}"
    inside = (s >= cfg.spread_min_bps - TOL) & (s <= cfg.spread_max_bps + TOL)
    assert inside.all(), f"spreads must lie in [spread_min_bps, spread_max_bps]: {s.to_dict()}"
    assert (np.diff(s.to_numpy()) <= TOL).all(), f"more $ volume -> no wider spread: {s}"
    assert s.iloc[0] > s.iloc[-1], "a $100k/day stock must cost more than a $10bn/day one"

    gaps = fn(pd.Series({1: np.nan, 2: 0.0, 3: 1e5}), cfg)
    assert np.isfinite(gaps.to_numpy()).all(), "missing / zero $ volume must get a spread"
    assert (gaps[[1, 2]] >= gaps[3] - TOL).all(), (
        f"missing / zero $ volume must not be cheaper than the least liquid stock: {gaps}"
    )


def _turnover_from(w: pd.Series, previous: pd.Series) -> float:
    union = w.index.union(previous.index, sort=False)
    return float((w.reindex(union).fillna(0.0) - previous.reindex(union).fillna(0.0)).abs().sum())


def check_smooth_target(fn: SmoothTargetFn) -> None:
    prev = pd.Series({1: 0.2, 2: -0.2, 3: 0.1, "SPY": -0.1})
    tgt = pd.Series({1: 0.1, 2: -0.3, 4: 0.2, "XLK": -0.1})

    off = fn(tgt, prev, TurnoverConfig(smoothing_halflife=0.0))
    union = tgt.index.union(off.index, sort=False)
    assert np.allclose(off.reindex(union).fillna(0.0), tgt.reindex(union).fillna(0.0)), (
        f"smoothing_halflife 0 must return the target: {off.to_dict()}"
    )
    slow = fn(tgt, prev, TurnoverConfig(smoothing_halflife=1.0))
    slower = fn(tgt, prev, TurnoverConfig(smoothing_halflife=4.0))

    for name, w in (("halflife 1", slow), ("halflife 4", slower)):
        assert w.index.is_unique and np.isfinite(w.to_numpy()).all(), f"{name}: {w}"

    base, t1, t4 = (_turnover_from(w, prev) for w in (tgt, slow, slower))
    assert t1 < base - TOL, f"smoothing must trade less than the raw target ({t1} vs {base})"
    assert t4 <= t1 + TOL, f"a longer half-life must not trade more ({t4} vs {t1})"

    first = fn(tgt, pd.Series(dtype=float), TurnoverConfig(smoothing_halflife=2.0))
    assert np.isfinite(first.to_numpy()).all(), "first rebalance (no previous) must work"


def _scored_context(scores: dict[int, float]) -> RebalanceContext:
    positions = {
        i: (LONG if s < 0 else SHORT) if np.isfinite(s) else FLAT for i, s in scores.items()
    }
    ctx = make_context(positions)
    return dataclasses.replace(ctx, scores=pd.Series(scores, dtype=float))


def check_expected_alpha(fn: AlphaFn) -> None:
    ctx = _scored_context({1: -2.0, 2: -1.0, 3: 0.5, 4: 2.0, 5: np.nan})
    a = fn(ctx, ResearchConfig())
    scored = a.reindex([1, 2, 3, 4])
    assert np.isfinite(scored.to_numpy()).all(), f"scored stocks need an alpha: {a.to_dict()}"
    assert (np.diff(scored.to_numpy()) < 0).all(), f"alpha must fall as s rises: {a.to_dict()}"
    assert scored[1] > 0 > scored[4], "s = -2 -> expected gain, s = +2 -> expected loss"
    assert abs(float(a.get(5, 0.0))) < TOL, "an unscored stock must get no alpha"


def check_residual_covariance(fn: CovarianceFn) -> None:
    ctx = make_context({i: LONG for i in range(1, 9)})
    pca = PcaResidualModel(k=2).fit(ctx.returns, pd.DataFrame(index=ctx.returns.index))
    ctx = dataclasses.replace(ctx, fitted=pca)
    cov = fn(ctx)
    ids = set(ctx.returns.columns)
    assert set(cov.index) == ids and set(cov.columns) == ids, "one row/column per stock"
    m = cov.loc[list(ctx.returns.columns), list(ctx.returns.columns)].to_numpy(dtype=float)
    assert np.allclose(m, m.T), "covariance must be symmetric"
    assert np.linalg.eigvalsh(m).min() >= -1e-12, "covariance must be positive semi-definite"
    assert (np.diag(m) > 0).all(), "every stock needs a positive residual variance"


def _optimizer_config(**optimizer: float) -> ResearchConfig:
    return ResearchConfig.model_validate(
        {
            "portfolio": {"max_weight": 0.5, "gross_leverage": 2.0, "dollar_neutral": False},
            "optimizer": optimizer,
        }
    )


def check_optimize_weights(fn: OptimizeFn) -> None:
    ids = [1, 2, 3, 4]
    alpha = pd.Series({1: 0.002, 2: -0.002, 3: 0.001, 4: -0.001})
    cov = pd.DataFrame(np.diag([1e-4] * 4), index=ids, columns=ids)
    dv = pd.Series(1e8, index=ids)
    none = pd.Series(dtype=float)
    cfg = _optimizer_config()

    w = fn(alpha, cov, none, dv, cfg).reindex(ids).fillna(0.0)
    assert (w.abs() <= 0.5 + 1e-6).all(), f"|w| must respect max_weight 0.5: {w.to_dict()}"
    assert w.abs().sum() <= 2.0 + 1e-6, f"sum |w| = {w.abs().sum():.4f} > gross_leverage 2"
    held = w[w.abs() > 1e-6]
    assert {1, 2} <= set(held.index), f"the strongest alphas must be held: {w.to_dict()}"
    assert (np.sign(held) == np.sign(alpha[held.index])).all(), "weights must follow alpha"

    zero = fn(pd.Series(0.0, index=ids), cov, none, dv, cfg).reindex(ids).fillna(0.0)
    assert (zero.abs() < 1e-6).all(), f"no alpha and nothing held -> no book: {zero.to_dict()}"

    prev = pd.Series({1: -0.3, 2: 0.3})  # opposite to alpha: costly to reverse
    free = fn(alpha, cov, prev, dv, _optimizer_config(cost_aversion=0.0))
    some = fn(alpha, cov, prev, dv, _optimizer_config(cost_aversion=5.0))
    stuck = fn(alpha, cov, prev, dv, _optimizer_config(cost_aversion=1e6))
    t0, t5, t_inf = (_turnover_from(x, prev) for x in (free, some, stuck))
    assert t5 <= t0 + 1e-6, f"more cost aversion must not trade more ({t5} vs {t0})"
    assert t_inf < 0.05 * t0, f"prohibitive cost aversion must keep the book ({t_inf} vs {t0})"


def check_optimized_target_weights(fn: OptimizedTargetFn) -> None:
    ctx = _scored_context({1: -2.0, 2: -1.5, 3: -1.8, 4: 1.6, 5: 2.0})
    cfg = ResearchConfig.model_validate(
        {
            "portfolio": {
                "gross_leverage": 2.0,
                "max_weight": 0.5,
                "hedge": True,
                "dollar_neutral": True,
                "construction": "optimizer",
            }
        }
    )
    w = _held(fn(ctx, pd.Series(dtype=float), cfg))
    assert not w.empty, "scored stocks with alpha must produce a book"
    assert w.abs().sum() <= 2.0 + 1e-6, f"sum |w| = {w.abs().sum():.4f} > gross_leverage 2"
    assert abs(w.sum()) < 1e-6, f"dollar_neutral: sum w = {w.sum():.3g}, expected 0"
    assert any(isinstance(i, str) for i in w.index), "hedge=True must hold the ETF hedges"

    unscored = _scored_context({1: np.nan, 2: np.nan})
    assert _held(fn(unscored, pd.Series(dtype=float), cfg)).empty, "no scores -> nothing held"
