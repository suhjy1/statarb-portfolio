from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from statarb.models.ou import TRADING_DAYS, OUParams

REPO_ROOT = Path(__file__).resolve().parents[3]


def cpp_module() -> ModuleType:
    try:
        import statarb_cpp
    except ImportError:
        build = str(REPO_ROOT / "build/cpp/python")

        if build not in sys.path:
            sys.path.append(build)

        try:
            import statarb_cpp
        except ImportError as exc:
            raise ImportError(
                "statarb_cpp is not built (see README: Optional C++ kernels)"
            ) from exc
    return statarb_cpp


def _fits(paths: np.ndarray, dt: float) -> list[OUParams | None]:
    out = cpp_module().fit_ou_batch(np.ascontiguousarray(paths, dtype=float), dt)
    return [
        OUParams(kappa=float(k), m=float(m), sigma=float(s)) if ok else None
        for k, m, s, ok in zip(out["kappa"], out["m"], out["sigma"], out["valid"], strict=True)
    ]


def fit_ou_cpp(x: np.ndarray, dt: float = 1.0 / TRADING_DAYS) -> OUParams | None:
    return _fits(np.asarray(x, dtype=float).reshape(-1, 1), dt)[0]


def fit_ou_columns(
    paths: pd.DataFrame, dt: float = 1.0 / TRADING_DAYS
) -> dict[object, OUParams | None]:
    if paths.shape[1] == 0:
        return {}
    return dict(zip(paths.columns, _fits(paths.to_numpy(dtype=float), dt), strict=True))
