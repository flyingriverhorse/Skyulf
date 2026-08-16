"""Tests for skyulf.modeling._evaluation.common helpers."""

from typing import cast

import numpy as np
import pytest

from skyulf.modeling._evaluation.common import (
    _align_thresholds,
    _downsample_indices,
    _sanitize_structure,
    downsample_curve,
    sanitize_metrics,
)

# ---------------------------------------------------------------------------
# _is_finite_number
# ---------------------------------------------------------------------------


def test_is_finite_number_float_finite():
    """Regular float should be considered finite."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number(3.14) is True


def test_is_finite_number_float_nan():
    """NaN float should not be considered finite."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number(float("nan")) is False


def test_is_finite_number_float_inf():
    """Infinity should not be considered finite."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number(float("inf")) is False


def test_is_finite_number_int():
    """Integers are always finite."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number(42) is True


def test_is_finite_number_numpy_float():
    """numpy.float64 values should be handled correctly."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number(np.float64(1.5)) is True
    assert _is_finite_number(np.float64(np.nan)) is False


def test_is_finite_number_numpy_int():
    """numpy integers are always finite."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number(np.int32(7)) is True


def test_is_finite_number_string_returns_false():
    """Strings are not numbers so should return False."""
    from skyulf.modeling._evaluation.common import _is_finite_number

    assert _is_finite_number("hello") is False


# ---------------------------------------------------------------------------
# sanitize_metrics
# ---------------------------------------------------------------------------


def test_sanitize_metrics_clean_dict():
    """All-finite dict should pass through unchanged."""
    metrics = {"accuracy": 0.95, "f1": 0.90}
    result = sanitize_metrics(metrics)
    assert result == {"accuracy": 0.95, "f1": 0.90}


def test_sanitize_metrics_removes_nan():
    """NaN values should be removed from the output."""
    metrics = {"accuracy": 0.9, "f1": float("nan")}
    result = sanitize_metrics(metrics)
    assert "f1" not in result
    assert result["accuracy"] == pytest.approx(0.9)


def test_sanitize_metrics_removes_inf():
    """Infinite values should be removed from the output."""
    metrics = {"r2": float("inf"), "mse": 0.1}
    result = sanitize_metrics(metrics)
    assert "r2" not in result
    assert result["mse"] == pytest.approx(0.1)


def test_sanitize_metrics_empty_dict():
    """Empty metrics dict should return empty dict."""
    assert sanitize_metrics({}) == {}


def test_sanitize_metrics_all_nan():
    """All-NaN dict should return empty dict."""
    metrics = {"a": float("nan"), "b": float("inf")}
    result = sanitize_metrics(metrics)
    assert result == {}


def test_sanitize_metrics_converts_numpy_float():
    """numpy floats should be cast to plain Python floats."""
    metrics = {"score": np.float64(0.75)}
    result = sanitize_metrics(cast(dict[str, float], metrics))
    assert isinstance(result["score"], float)
    assert result["score"] == pytest.approx(0.75)


def test_sanitize_structure_handles_nested_list():
    """A list value must be recursively sanitized and rebuilt as a list."""
    warnings: list = []
    result = _sanitize_structure([1, 2.5, float("nan")], warnings=warnings, context="ctx")
    assert result == [1, 2.5, None]
    assert isinstance(result, list)


def test_sanitize_structure_handles_nested_tuple():
    """A tuple value must be recursively sanitized and rebuilt as a tuple."""
    warnings: list = []
    result = _sanitize_structure((1, 2, 3), warnings=warnings, context="ctx")
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_sanitize_structure_int_returns_plain_int():
    """A finite Python/numpy int must be cast to a plain int (not float)."""
    warnings: list = []
    result = _sanitize_structure(np.int64(7), warnings=warnings, context="ctx")
    assert result == 7
    assert isinstance(result, int)


def test_sanitize_structure_non_numeric_passthrough():
    """Non-numeric, non-container values (e.g. strings) must be returned unchanged."""
    warnings: list = []
    result = _sanitize_structure("hello", warnings=warnings, context="ctx")
    assert result == "hello"


# ---------------------------------------------------------------------------
# _downsample_indices
# ---------------------------------------------------------------------------


def test_downsample_indices_below_limit():
    """When length <= limit all indices should be returned."""
    idx = _downsample_indices(5, 10)
    np.testing.assert_array_equal(idx, np.arange(5))


def test_downsample_indices_above_limit():
    """When length > limit exactly `limit` indices should be returned."""
    idx = _downsample_indices(1000, 100)
    assert len(idx) == 100
    # Must be sorted and within bounds
    assert idx[0] == 0
    assert idx[-1] == 999


def test_downsample_indices_exact_limit():
    """When length == limit result should be identical to arange."""
    idx = _downsample_indices(10, 10)
    np.testing.assert_array_equal(idx, np.arange(10))


# ---------------------------------------------------------------------------
# _align_thresholds
# ---------------------------------------------------------------------------


def test_align_thresholds_already_correct_size():
    """Thresholds already matching target size should be returned as-is."""
    t = np.array([0.1, 0.5, 0.9])
    result = _align_thresholds(t, 3)
    np.testing.assert_array_equal(result, t)


def test_align_thresholds_one_short():
    """When thresholds is one shorter than target the last value should be appended."""
    t = np.array([0.1, 0.5])
    result = _align_thresholds(t, 3)
    assert len(result) == 3
    assert result[-1] == pytest.approx(0.5)


def test_align_thresholds_too_long():
    """When thresholds is longer than target it should be trimmed."""
    t = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = _align_thresholds(t, 3)
    assert len(result) == 3


def test_align_thresholds_empty():
    """Empty thresholds should return zeros of target size."""
    t = np.array([])
    result = _align_thresholds(t, 4)
    assert len(result) == 4
    assert all(v == 0.0 for v in result)


def test_align_thresholds_needs_padding():
    """When thresholds are shorter (not by 1) last value should pad to target."""
    t = np.array([0.3, 0.6])
    result = _align_thresholds(t, 5)
    assert len(result) == 5
    assert result[-1] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# downsample_curve
# ---------------------------------------------------------------------------


def test_downsample_curve_returns_list_of_curve_points():
    """downsample_curve should return a list of CurvePoint objects."""
    from skyulf.modeling._evaluation.schemas import CurvePoint

    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    points = downsample_curve(x, y, limit=10)
    assert isinstance(points, list)
    assert all(isinstance(p, CurvePoint) for p in points)


def test_downsample_curve_respects_limit():
    """Point count should not exceed limit when input is large."""
    x = np.linspace(0, 1, 2000)
    y = np.linspace(0, 1, 2000)
    points = downsample_curve(x, y, limit=100)
    assert len(points) <= 100


def test_downsample_curve_small_input():
    """Small inputs should not be expanded beyond their size."""
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    points = downsample_curve(x, y, limit=1000)
    assert len(points) == 3
