from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPDR_SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CleaningConfig(_Strict):
    use_excess_returns: bool = True
    winsorize_n_mad: float | None = Field(
        default=None, gt=0, description="Cross-sectional clip at median ± n·MAD; None = off."
    )
    min_obs_fraction: float = Field(
        default=0.9, gt=0, le=1, description="Share of a window a stock must have data for."
    )


class ModelConfig(_Strict):
    kind: Literal["etf", "pca"] = "pca"
    estimation_window: int = Field(default=252, gt=20, description="Days for betas/loadings.")
    etfs: tuple[str, ...] = ("SPY", *SPDR_SECTOR_ETFS)
    pca_k: int = Field(default=15, gt=0, description="Number of eigenportfolios.")
    drop_alpha: bool = Field(
        default=True, description="Subtract the in-window intercept from residuals (A&L)."
    )
    etf_fallback: str = Field(
        default="SPY", description="ETF used unless another fits clearly better."
    )
    etf_min_r2_gain: float = Field(
        default=0.0, ge=0, description="R² margin a non-fallback ETF must beat it by."
    )
    etf_split_halves: bool = Field(
        default=False, description="Require the R² margin in both halves of the window too."
    )
    etf_switch_margin: float = Field(
        default=0.02, ge=0, description="Keep last refit's ETF unless beaten by this R²; 0 = off."
    )


class OUConfig(_Strict):
    window: int = Field(default=60, gt=10, description="Days of cumulative residual to fit.")
    min_kappa: float = Field(
        default=252 / 30, ge=0, description="Annualised speed; 252/30 = reversion < ~30d; 0 = off."
    )


class SignalConfig(_Strict):
    open_long: float = -1.25
    open_short: float = 1.25
    close_long: float = -0.50
    close_short: float = 0.75
    center_m: bool = Field(
        default=False,
        description="A&L eq. 15-16: subtract the cross-sectional mean of m before scoring.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> SignalConfig:
        if not (self.open_long < self.close_long and self.close_short < self.open_short):
            raise ValueError("need open_long < close_long and close_short < open_short")
        return self


class PortfolioConfig(_Strict):
    sizing: Literal["equal", "inverse_vol"] = "equal"
    vol_window: int = Field(default=60, gt=1, description="Days of returns for inverse_vol.")
    gross_leverage: float = Field(default=2.0, gt=0, description="Target sum |w|, all legs.")
    max_weight: float = Field(default=0.02, gt=0, le=1, description="Cap on |w| per stock.")
    hedge: bool = Field(default=True, description="Offset factor/ETF exposure of stock legs.")
    dollar_neutral: bool = Field(default=True, description="Make the whole book sum to 0.")
    construction: Literal["rule", "optimizer"] = Field(
        default="rule",
        description="rule: construction.target_weights; optimizer: optimizer."
        "optimized_target_weights (see OptimizerConfig).",
    )


class TurnoverConfig(_Strict):
    smoothing_halflife: float = Field(
        default=0.0,
        ge=0,
        description="Blend new targets with the previous rebalance's; half-life in "
        "rebalances; 0 = off.",
    )


class OptimizerConfig(_Strict):
    risk_aversion: float = Field(default=1.0, gt=0, description="Penalty on predicted variance.")
    cost_aversion: float = Field(
        default=1.0, ge=0, description="Multiplier on predicted trading cost (1 = as modelled)."
    )
    horizon: int = Field(
        default=1,
        ge=1,
        description="Rebalances in the receding-horizon plan; execute only its first target.",
    )
    alpha_scale: float = Field(
        default=0.001,
        gt=0,
        description="Return per negative s-score per rebalance when OU fits "
        "are unavailable (hand-built contexts only).",
    )
    alpha_decay: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Fraction of the initial alpha forecast remaining each future rebalance.",
    )
    alpha_calibration: Literal["none", "online"] = "none"
    calib_window_rebalances: int = Field(default=52, gt=0)
    calib_min_rebalances: int = Field(default=26, gt=0)

    @model_validator(mode="after")
    def _calibration_history(self) -> OptimizerConfig:
        if self.calib_min_rebalances > self.calib_window_rebalances:
            raise ValueError("calib_min_rebalances must not exceed calib_window_rebalances")
        return self


class CostConfig(_Strict):
    stock_half_spread_bps: float = Field(default=5.0, ge=0, description="Per side (A&L: 5).")
    etf_half_spread_bps: float = Field(default=1.0, ge=0, description="Per side, ETF hedges.")
    impact_coef: float = Field(default=0.0, ge=0, description="Market impact scale; 0 = off.")
    borrow_bps_annual: float = Field(default=50.0, ge=0, description="Fee on short notional.")
    capital: float = Field(default=1e7, gt=0, description="Book size in $ (for impact vs ADV).")
    spread_model: Literal["constant", "liquidity"] = Field(
        default="constant",
        description="constant: stock_half_spread_bps for every stock; liquidity: "
        "costs.stock_half_spread_bps from $ volume.",
    )
    spread_ref_dollar_volume: float = Field(
        default=1e8, gt=0, description="Liquidity model: $ volume that pays stock_half_spread_bps."
    )
    spread_elasticity: float = Field(
        default=0.5, ge=0, description="Liquidity model: how fast spreads widen as $ volume falls."
    )
    spread_min_bps: float = Field(default=1.0, ge=0, description="Liquidity model floor.")
    spread_max_bps: float = Field(default=100.0, ge=0, description="Liquidity model ceiling.")

    @model_validator(mode="after")
    def _spread_bounds(self) -> CostConfig:
        if self.spread_min_bps > self.spread_max_bps:
            raise ValueError("need spread_min_bps <= spread_max_bps")
        return self


class BacktestConfig(_Strict):
    execution_lag: int = Field(
        default=1,
        ge=1,
        description=(
            "Targets from data through close t first earn day t+lag's return: 1 = filled at "
            "the close of t (optimistic, A&L), 2 = filled at the close of t+1."
        ),
    )
    min_trade: float = Field(
        default=0.0,
        ge=0,
        description="No-trade band: skip trades smaller than this fraction of capital "
        "(full exits to 0 always trade); 0 = trade every difference.",
    )


class PeriodConfig(_Strict):
    start: dt.date
    end: dt.date


class ResearchConfig(_Strict):
    rebalance_every: int = Field(default=1, gt=0, description="Trading days between refits.")
    warmup_days: int | None = Field(
        default=None, description="History before the start date; default = longest window."
    )
    cleaning: CleaningConfig = CleaningConfig()
    model: ModelConfig = ModelConfig()
    ou: OUConfig = OUConfig()
    signal: SignalConfig = SignalConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    turnover: TurnoverConfig = TurnoverConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    costs: CostConfig = CostConfig()
    backtest: BacktestConfig = BacktestConfig()
    period: PeriodConfig | None = None

    @property
    def required_warmup(self) -> int:
        return self.warmup_days or self.model.estimation_window + self.ou.window
