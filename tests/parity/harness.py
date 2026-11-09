from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ATOL = 1e-10


@dataclass(frozen=True)
class ParityResult:
    max_abs_diff: float
    atol: float
    nan_mismatch: int
    shape: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return self.nan_mismatch == 0 and self.max_abs_diff <= self.atol

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status}: max|diff|={self.max_abs_diff:.3e} (atol={self.atol:.1e}), "
            f"nan mismatches={self.nan_mismatch}, shape={self.shape}"
        )


def _as_array(x: Any) -> np.ndarray:
    if isinstance(x, pd.DataFrame | pd.Series):
        x = x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def compare(
    python_fn: Callable[..., Any],
    cpp_fn: Callable[..., Any],
    dataset: Mapping[str, Any],
    *,
    atol: float = DEFAULT_ATOL,
) -> ParityResult:
    expected = _as_array(python_fn(**dataset))
    actual = _as_array(cpp_fn(**dataset))

    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch: python {expected.shape} vs cpp {actual.shape}")

    nan_e, nan_a = np.isnan(expected), np.isnan(actual)
    both = ~(nan_e | nan_a)
    diff = np.abs(expected[both] - actual[both])

    return ParityResult(
        max_abs_diff=float(diff.max()) if diff.size else 0.0,
        atol=atol,
        nan_mismatch=int((nan_e != nan_a).sum()),
        shape=expected.shape,
    )


def assert_parity(
    python_fn: Callable[..., Any],
    cpp_fn: Callable[..., Any],
    dataset: Mapping[str, Any],
    *,
    atol: float = DEFAULT_ATOL,
) -> ParityResult:
    result = compare(python_fn, cpp_fn, dataset, atol=atol)
    assert result.passed, str(result)
    return result
