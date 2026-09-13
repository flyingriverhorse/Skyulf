"""Keep ManualBounds targets aligned when some configured columns are absent."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines import EngineRegistry
from skyulf.preprocessing.outliers.manual_bounds import (
    ManualBoundsApplier,
    ManualBoundsCalculator,
)


@pytest.fixture(
    params=[("pandas", False), ("pandas", True), ("polars", False), ("polars", True)],
    ids=["pandas", "pandas-wrapped", "polars", "polars-wrapped"],
)
def frame_kind(request):
    """Exercise native frames and the public Skyulf wrappers on both engines."""
    return request.param


def _frame(frame_kind, values):
    """Keep missing values native so Polars null and NaN remain distinguishable."""
    engine, wrapped = frame_kind
    frame = (
        pd.DataFrame({"age": values}, dtype=float)
        if engine == "pandas"
        else pl.DataFrame({"age": values}, schema={"age": pl.Float64})
    )
    return EngineRegistry.wrap(frame) if wrapped else frame


def _target(frame_kind, shape, values):
    """Build each accepted target shape without introducing mixed-engine pairs."""
    engine, _wrapped = frame_kind
    if shape == "list":
        return values
    if shape == "numpy":
        return np.asarray(values)
    if shape == "frame":
        return (
            pd.DataFrame({"target": values})
            if engine == "pandas"
            else pl.DataFrame({"target": values})
        )
    return pd.Series(values, name="target") if engine == "pandas" else pl.Series("target", values)


def _native(frame):
    """Use the public accessor when comparing wrapped results."""
    return frame.to_native() if hasattr(frame, "to_native") else frame


@pytest.mark.parametrize("target_shape", ["series", "frame", "numpy", "list"])
@pytest.mark.parametrize("row_count", [0, 3], ids=["empty", "nonempty"])
def test_missing_only_bounds_preserve_all_feature_and_target_rows(
    frame_kind, target_shape, row_count
):
    """Ignoring an absent column must never broadcast a one-row mask onto a multi-row target."""
    ages = [18.0, 40.0, 65.0][:row_count]
    labels = [0, 10, 20][:row_count]
    features = _frame(frame_kind, ages)
    target = _target(frame_kind, target_shape, labels)
    artifact = ManualBoundsCalculator().fit(
        (features, target), {"bounds": {"target": {"upper": 10}}}
    )

    actual_features, actual_target = ManualBoundsApplier().apply((features, target), artifact)

    assert type(actual_target) is type(target)
    np.testing.assert_array_equal(np.asarray(actual_target).ravel(), labels)
    assert _native(actual_features)["age"].to_list() == ages
    assert len(actual_features) == len(actual_target) == row_count


@pytest.mark.parametrize("row_count", [0, 3], ids=["empty", "nonempty"])
def test_empty_bounds_preserve_feature_and_target_rows(frame_kind, row_count):
    """An empty bounds mapping must retain the existing no-op behavior for X and y."""
    ages = [18.0, 40.0, 65.0][:row_count]
    features = _frame(frame_kind, ages)
    target = _target(frame_kind, "series", [0, 10, 20][:row_count])

    actual_features, actual_target = ManualBoundsApplier().apply((features, target), {"bounds": {}})

    np.testing.assert_array_equal(np.asarray(actual_target).ravel(), np.asarray(target).ravel())
    assert _native(actual_features)["age"].to_list() == ages
    assert len(actual_features) == len(actual_target) == row_count


@pytest.mark.parametrize("missing_first", [False, True])
def test_missing_bounds_do_not_change_valid_filter_or_missing_value_alignment(
    frame_kind, missing_first
):
    """Ignored columns must not change inclusive filtering or target alignment for null/NaN rows."""
    features = _frame(frame_kind, [17.0, 18.0, 65.0, 66.0, None, np.nan])
    target = _target(frame_kind, "series", [0, 10, 20, 30, 40, 50])
    items = [("age", {"lower": 18, "upper": 65}), ("removed", {"lower": 1000})]
    bounds = dict(reversed(items) if missing_first else items)

    actual_features, actual_target = ManualBoundsApplier().apply(
        (features, target), {"bounds": bounds}
    )

    np.testing.assert_allclose(
        np.asarray(_native(actual_features)["age"]), [18.0, 65.0, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(np.asarray(actual_target).ravel(), [10, 20, 40, 50])
    assert len(actual_features) == len(actual_target) == 4
