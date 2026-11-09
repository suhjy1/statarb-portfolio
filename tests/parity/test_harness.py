import numpy as np
import pandas as pd
import pytest

from parity.harness import assert_parity, compare


def test_identical_outputs_pass() -> None:
    data = {"x": np.arange(5.0)}
    result = compare(lambda x: x * 2, lambda x: x + x, data)
    assert result.passed
    assert result.max_abs_diff == 0.0


def test_difference_above_tolerance_fails() -> None:
    data = {"x": np.zeros(3)}
    result = compare(lambda x: x, lambda x: x + 1e-6, data, atol=1e-8)
    assert not result.passed
    assert result.max_abs_diff == pytest.approx(1e-6)

    with pytest.raises(AssertionError, match="FAIL"):
        assert_parity(lambda x: x, lambda x: x + 1e-6, data, atol=1e-8)


def test_nan_mask_mismatch_fails() -> None:
    data = {"x": np.array([1.0, np.nan, 3.0])}
    assert compare(lambda x: x, lambda x: x, data).passed
    result = compare(lambda x: x, lambda x: np.nan_to_num(x), data)
    assert result.nan_mismatch == 1
    assert not result.passed


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare(lambda: np.zeros(2), lambda: np.zeros(3), {})


def test_accepts_pandas_and_scalars() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert compare(lambda: frame, lambda: frame.to_numpy(), {}).shape == (2, 2)
    assert compare(lambda: 1.0, lambda: 1.0 + 1e-12, {}).passed
