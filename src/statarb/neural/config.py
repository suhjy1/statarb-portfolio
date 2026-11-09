from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModernConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(default="modern_daily_v1", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    architecture: Literal["linear", "lstm", "conv_transformer"] = "conv_transformer"
    train_start: dt.date = dt.date(2010, 1, 4)
    train_end: dt.date = dt.date(2017, 12, 29)
    test_start: dt.date = dt.date(2018, 1, 2)
    test_end: dt.date = dt.date(2026, 10, 16)
    lookback: int = Field(default=30, ge=3)
    hidden_size: int = Field(default=16, ge=4)
    max_stocks: int | None = Field(default=None, ge=3)
    training_steps: int = Field(default=200, gt=0)
    training_window_days: int = Field(default=1000, ge=10)
    refit_every: int = Field(default=125, gt=0)
    chunk_days: int = Field(default=8, ge=3)
    learning_rate: float = Field(default=0.001, gt=0)
    risk_aversion: float = Field(default=1.0, ge=0)
    seed: int = 42
    cpu_threads: int = Field(default=1, gt=0)
    borrow_bps_annual: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _chronology(self) -> ModernConfig:
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("require train_start <= train_end < test_start <= test_end")

        if self.hidden_size % 2:
            raise ValueError("hidden_size must be even for attention heads")

        return self
