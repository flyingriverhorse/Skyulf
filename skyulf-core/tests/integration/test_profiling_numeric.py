"""Tests for skyulf.profiling._analyzer.numeric.NumericMixin._calculate_vif.

Covers the branches not exercised by test_profiling_analyzer.py's happy-path
VIF tests: a constant column producing a NaN correlation, a perfectly
collinear (singular) correlation matrix, and the generic exception fallback.
"""

from typing import Any

import numpy as np
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.analyzer import EDAAnalyzer


def test_calculate_vif_returns_none_for_constant_column() -> None:
    """A constant numeric column yields NaN in the correlation matrix -> None (line 51)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    constant = np.full(50, 5.0)
    df = pl.DataFrame({"a": a, "constant": constant})
    analyzer = EDAAnalyzer(df)

    result = analyzer._calculate_vif(["a", "constant"])

    assert result is None


def test_calculate_vif_singular_matrix_flags_all_columns() -> None:
    """Perfectly collinear columns (b = 2*a exactly) make the corr matrix singular (lines 55-57)."""
    a = np.linspace(1.0, 50.0, 50)
    b = a * 2.0
    df = pl.DataFrame({"a": a, "b": b})
    analyzer = EDAAnalyzer(df)

    result = analyzer._calculate_vif(["a", "b"])

    assert result == {"a": 999.0, "b": 999.0}


@pytest.mark.parametrize("noise", [0.5, 1e-5, 1e-7])
@pytest.mark.parametrize("rescaled", [False, True])
def test_calculate_vif_matches_known_orthogonal_design(noise: float, rescaled: bool) -> None:
    """Near-collinear features need accurate VIF without blaming an orthogonal feature."""
    a = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    independent_noise = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    data = np.column_stack((a, 2 * a + noise * independent_noise, a * independent_noise))
    if rescaled:
        data = data * [1e-6, -1000.0, 7.0] + [3.0, 500.0, 29.0]
    frame = pl.DataFrame(data, schema=["a", "b", "independent"])

    result = EDAAnalyzer(frame)._calculate_vif(frame.columns)

    # Var(a)=Var(noise)=1 and Cov(a,noise)=0 give VIF = 1 + 4 / noise**2.
    expected = 1.0 + 4.0 / noise**2
    assert result is not None
    assert result["a"] == pytest.approx(expected, rel=1e-5)
    assert result["b"] == pytest.approx(expected, rel=1e-5)
    assert result["independent"] == pytest.approx(1.0)


def test_calculate_vif_singular_design_preserves_independent_feature() -> None:
    """A duplicated pair must not cause a false removal warning for an unrelated feature."""
    a = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    independent = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    frame = pl.DataFrame({"a": a, "b": 2 * a, "independent": independent})

    result = EDAAnalyzer(frame)._calculate_vif(frame.columns)

    assert result is not None
    assert result["a"] > 10.0
    assert result["b"] > 10.0
    assert result["independent"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid_vif", [-1e22, 0.5, np.nan, np.inf])
def test_calculate_vif_recovers_from_invalid_inverse_diagonal(
    monkeypatch: pytest.MonkeyPatch, invalid_vif: float
) -> None:
    """Numerically invalid inverse values must not become an all-clear or non-finite report."""
    a = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    noise = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    frame = pl.DataFrame({"a": a, "b": 2 * a + 0.5 * noise})

    def invalid_inverse(matrix: np.ndarray) -> np.ndarray:
        """Simulate a numerical failure while retaining the real input design."""
        return np.diag([invalid_vif, invalid_vif])

    monkeypatch.setattr(np.linalg, "inv", invalid_inverse)

    result = EDAAnalyzer(frame)._calculate_vif(frame.columns)

    assert result == pytest.approx({"a": 17.0, "b": 17.0})


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_near_collinearity_produces_public_vif_alerts(seed: int) -> None:
    """An unstable inverse must not turn almost duplicated features into an all-clear."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=100)
    frame = pl.DataFrame(
        {"a": a, "b": 2 * a + 1e-9 * rng.normal(size=100), "independent": rng.normal(size=100)}
    )

    profile = EDAAnalyzer(frame).analyze()

    assert profile.vif is not None
    assert all(np.isfinite(value) for value in profile.vif.values())
    assert profile.vif["a"] > 10.0
    assert profile.vif["b"] > 10.0
    assert profile.vif["independent"] < 5.0
    messages = [alert.message for alert in profile.alerts if alert.type == "Multicollinearity"]
    assert len(messages) == 2
    assert any("'a'" in message for message in messages)
    assert any("'b'" in message for message in messages)


def test_calculate_vif_generic_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected exception (e.g. from np.corrcoef) should be caught and return None (lines 60-62)."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0, 1, 50)
    df = pl.DataFrame({"a": a, "b": b})
    analyzer = EDAAnalyzer(df)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("corrcoef exploded")

    monkeypatch.setattr(np, "corrcoef", _boom)

    result = analyzer._calculate_vif(["a", "b"])

    assert result is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``age``/``income`` rows — closer to production data than
    the small synthetic frames used elsewhere in this file.
    """

    def test_calculate_vif_on_customers_drops_missing_rows(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)

        result = analyzer._calculate_vif(["age", "income"])

        # 15 rows minus the ~5 rows with a missing age/income leaves 10, which
        # clears the "len(numeric_cols) + 5" floor, so real VIF values come back.
        assert result is not None
        assert set(result.keys()) == {"age", "income"}
        assert all(v >= 1.0 for v in result.values())
