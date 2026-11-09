from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from parity.harness import assert_parity
from statarb.models.ou import OUParams, fit_ou, simulate_ou

if os.environ.get("STATARB_REQUIRE_CPP") == "1":
    import statarb_cpp  # noqa: F401
else:
    pytest.importorskip("statarb_cpp", reason="build the C++ kernels first (see README)")

from statarb.native.ou import fit_ou_columns, fit_ou_cpp


def _vec(p: OUParams | None) -> np.ndarray:
    return np.full(3, np.nan) if p is None else np.array([p.kappa, p.m, p.sigma])


@pytest.mark.parametrize("kappa", [5.0, 20.0, 60.0])
def test_fit_matches_python(kappa: float) -> None:
    x = simulate_ou(kappa, m=0.01, sigma=0.2, n=60, seed=int(kappa))
    assert_parity(lambda x: _vec(fit_ou(x)), lambda x: _vec(fit_ou_cpp(x)), {"x": x}, atol=1e-8)


@pytest.mark.parametrize(
    "x",
    [
        np.array([0.0, 0.1]),  # too short
        np.array([0.0, 0.1, np.nan, 0.2, 0.1]),  # gap
        1.05 ** np.arange(30.0),  # explosive trend: b >= 1
        np.tile([1.0, -1.0], 15),  # alternating: b <= 0
    ],
    ids=["short", "nan", "trend", "alternating"],
)
def test_rejections_match(x: np.ndarray) -> None:
    assert fit_ou(x) is None
    assert fit_ou_cpp(x) is None


def test_batch_matches_per_column() -> None:
    paths = pd.DataFrame(
        {sid: simulate_ou(10.0 + sid, m=0.0, sigma=0.15, n=60, seed=sid) for sid in range(50)}
    )
    paths.iloc[5, 3] = np.nan  # one invalid column must not affect the others
    batch = fit_ou_columns(paths)
    assert list(batch) == list(paths.columns)
    expected = np.vstack([_vec(fit_ou(paths[c].to_numpy())) for c in paths.columns])
    actual = np.vstack([_vec(batch[c]) for c in paths.columns])
    assert_parity(lambda: expected, lambda: actual, {}, atol=1e-8)


@pytest.mark.parametrize("dt", [1 / 252, 1 / 52, 1.0])
@pytest.mark.parametrize("offset", [0.0, 2.0, -3.0])
def test_offset_and_timestep_parity(dt: float, offset: float) -> None:
    x = simulate_ou(15, 0, 0.2, 100, seed=32) + offset
    np.testing.assert_allclose(_vec(fit_ou_cpp(x, dt)), _vec(fit_ou(x, dt)), atol=1e-8)


def test_constant_and_empty_inputs() -> None:
    for value in (0.0, 2.0):
        x = np.full(20, value)
        np.testing.assert_allclose(_vec(fit_ou_cpp(x)), _vec(fit_ou(x)), atol=1e-8, equal_nan=True)
    assert fit_ou_columns(pd.DataFrame()) == {}
