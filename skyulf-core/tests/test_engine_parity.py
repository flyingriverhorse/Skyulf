"""Property-based engine-parity tests.

For each covered Calculator, fitting on a pandas.DataFrame and on the
equivalent polars.DataFrame must yield numerically identical artifacts.

Backlog item E4 — second slice: imputation + outlier calculators added
alongside the first-slice scalers.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# Hypothesis profile: keep tight so the suite stays fast in CI.
settings.register_profile(
    "engine_parity",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("engine_parity")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Bounded floats: avoid inf/nan to keep parity meaningful (NaN handling is
# an orthogonal concern with its own dedicated tests).
_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _numeric_frame(draw: st.DrawFn, *, min_rows: int = 5, max_rows: int = 50) -> pd.DataFrame:
    """Generate a small numeric DataFrame with 2 columns (`a`, `b`)."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    a = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    b = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    return pd.DataFrame({"a": a, "b": b})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_artifacts_equal(pd_params: Dict[str, Any], pl_params: Dict[str, Any]) -> None:
    """Compare two params dicts, treating numeric lists as approx-equal."""
    assert set(pd_params.keys()) == set(
        pl_params.keys()
    ), f"Key mismatch: pandas={set(pd_params)} polars={set(pl_params)}"
    for key, pd_val in pd_params.items():
        pl_val = pl_params[key]
        if isinstance(pd_val, list) and pd_val and isinstance(pd_val[0], (int, float)):
            np.testing.assert_allclose(
                np.asarray(pd_val, dtype=float),
                np.asarray(pl_val, dtype=float),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"Numeric mismatch on key '{key}'",
            )
        else:
            assert pd_val == pl_val, f"Mismatch on key '{key}': {pd_val!r} vs {pl_val!r}"


def _assert_bounds_equal(
    pd_params: Dict[str, Any], pl_params: Dict[str, Any], *, bounds_key: str = "bounds"
) -> None:
    """Compare outlier-detector artifacts that use per-column bound dicts."""
    assert pd_params.get("type") == pl_params.get("type")
    pd_bounds: Dict[str, Any] = pd_params.get(bounds_key, {})
    pl_bounds: Dict[str, Any] = pl_params.get(bounds_key, {})
    assert set(pd_bounds.keys()) == set(
        pl_bounds.keys()
    ), f"Bounds column mismatch: pandas={set(pd_bounds)} polars={set(pl_bounds)}"
    for col, pd_b in pd_bounds.items():
        pl_b = pl_bounds[col]
        for stat_key in pd_b:
            np.testing.assert_allclose(
                float(pd_b[stat_key]),
                float(pl_b[stat_key]),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"Bound mismatch col={col!r} key={stat_key!r}",
            )


def _assert_fill_values_equal(pd_params: Dict[str, Any], pl_params: Dict[str, Any]) -> None:
    """Compare SimpleImputer fill_values dicts."""
    assert pd_params.get("type") == pl_params.get("type")
    assert pd_params.get("strategy") == pl_params.get("strategy")
    assert pd_params.get("columns") == pl_params.get("columns")
    pd_fv: Dict[str, Any] = pd_params.get("fill_values", {})
    pl_fv: Dict[str, Any] = pl_params.get("fill_values", {})
    assert set(pd_fv.keys()) == set(pl_fv.keys())
    for col in pd_fv:
        np.testing.assert_allclose(
            float(pd_fv[col]),
            float(pl_fv[col]),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"fill_value mismatch on col={col!r}",
        )


# ---------------------------------------------------------------------------
# Scaler tests (first slice)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "calculator_path,config",
    [
        ("skyulf.preprocessing.scaling:StandardScalerCalculator", {"columns": ["a", "b"]}),
        ("skyulf.preprocessing.scaling:MinMaxScalerCalculator", {"columns": ["a", "b"]}),
        ("skyulf.preprocessing.scaling:MaxAbsScalerCalculator", {"columns": ["a", "b"]}),
    ],
)
@given(df=_numeric_frame())
def test_scaler_fit_engine_parity(
    calculator_path: str, config: Dict[str, Any], df: pd.DataFrame
) -> None:
    """Pandas and polars inputs must produce identical fitted artifacts."""
    module_name, class_name = calculator_path.split(":")
    module = __import__(module_name, fromlist=[class_name])
    calc_cls = getattr(module, class_name)

    pd_params = calc_cls().fit(df, dict(config))
    pl_params = calc_cls().fit(pl.from_pandas(df), dict(config))

    _assert_artifacts_equal(pd_params, pl_params)


@given(df=_numeric_frame(min_rows=10, max_rows=80))
def test_robust_scaler_fit_engine_parity(df: pd.DataFrame) -> None:
    """RobustScaler uses quantile statistics — verify pandas/polars parity."""
    from skyulf.preprocessing.scaling import RobustScalerCalculator

    config: Dict[str, Any] = {"columns": ["a", "b"]}
    pd_params = RobustScalerCalculator().fit(df, dict(config))
    pl_params = RobustScalerCalculator().fit(pl.from_pandas(df), dict(config))

    _assert_artifacts_equal(pd_params, pl_params)


# ---------------------------------------------------------------------------
# Imputation tests (second slice)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["mean", "median"])
@given(df=_numeric_frame(min_rows=10, max_rows=60))
def test_simple_imputer_fit_engine_parity(strategy: str, df: pd.DataFrame) -> None:
    """SimpleImputer has genuine dual paths (polars native vs sklearn).

    Mean and median are computed differently in each branch; assert they
    produce fill_values within floating-point tolerance.
    """
    from skyulf.preprocessing.imputation import SimpleImputerCalculator

    # detect_numeric_columns(exclude_constant=True) skips constant columns;
    # skip the frame if either column has no variance to keep parity valid.
    assume(df["a"].nunique() > 1 and df["b"].nunique() > 1)

    config: Dict[str, Any] = {"columns": ["a", "b"], "strategy": strategy}
    pd_params = SimpleImputerCalculator().fit(df, dict(config))
    pl_params = SimpleImputerCalculator().fit(pl.from_pandas(df), dict(config))

    _assert_fill_values_equal(pd_params, pl_params)


# ---------------------------------------------------------------------------
# Outlier-detector tests (second slice)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "calculator_path,extra_config",
    [
        ("skyulf.preprocessing.outliers:IQRCalculator", {"multiplier": 1.5}),
        ("skyulf.preprocessing.outliers:ZScoreCalculator", {"threshold": 3.0}),
        (
            "skyulf.preprocessing.outliers:WinsorizeCalculator",
            {"lower_percentile": 5.0, "upper_percentile": 95.0},
        ),
    ],
)
@given(df=_numeric_frame(min_rows=10, max_rows=80))
def test_outlier_calculator_engine_parity(
    calculator_path: str, extra_config: Dict[str, Any], df: pd.DataFrame
) -> None:
    """Outlier calculators convert polars → pandas internally.

    Both engines must produce equal bounds/stats for the same underlying data.
    """
    module_name, class_name = calculator_path.split(":")
    module = __import__(module_name, fromlist=[class_name])
    calc_cls = getattr(module, class_name)

    config: Dict[str, Any] = {"columns": ["a", "b"], **extra_config}
    pd_params = calc_cls().fit(df, dict(config))
    pl_params = calc_cls().fit(pl.from_pandas(df), dict(config))

    bounds_key = "bounds" if "bounds" in pd_params else "stats"
    _assert_bounds_equal(pd_params, pl_params, bounds_key=bounds_key)


# ---------------------------------------------------------------------------
# No-op / short-circuit parity
# ---------------------------------------------------------------------------


def _empty_columns_returns_empty_for(calc_cls: type) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert calc_cls().fit(df, {"columns": ["nonexistent"]}) == {}
    assert calc_cls().fit(pl.from_pandas(df), {"columns": ["nonexistent"]}) == {}


@pytest.mark.parametrize(
    "calculator_path",
    [
        "skyulf.preprocessing.scaling:StandardScalerCalculator",
        "skyulf.preprocessing.scaling:MinMaxScalerCalculator",
        "skyulf.preprocessing.scaling:MaxAbsScalerCalculator",
        "skyulf.preprocessing.scaling:RobustScalerCalculator",
        "skyulf.preprocessing.imputation:SimpleImputerCalculator",
        "skyulf.preprocessing.outliers:IQRCalculator",
        "skyulf.preprocessing.outliers:ZScoreCalculator",
        "skyulf.preprocessing.outliers:WinsorizeCalculator",
    ],
)
def test_unknown_columns_short_circuits_in_both_engines(calculator_path: str) -> None:
    """No-op behavior must be identical across engines."""
    module_name, class_name = calculator_path.split(":")
    module = __import__(module_name, fromlist=[class_name])
    _empty_columns_returns_empty_for(getattr(module, class_name))
