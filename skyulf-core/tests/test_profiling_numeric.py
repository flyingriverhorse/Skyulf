"""Tests for skyulf.profiling._analyzer.numeric.NumericMixin._calculate_vif.

Covers the branches not exercised by test_profiling_analyzer.py's happy-path
VIF tests: a constant column producing a NaN correlation, a perfectly
collinear (singular) correlation matrix, and the generic exception fallback.
"""

from typing import Any

import numpy as np
import polars as pl
import pytest

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
