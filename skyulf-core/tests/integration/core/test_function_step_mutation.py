"""Project callbacks must preserve caller data and the original row alignment."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from skyulf.preprocessing import FeatureEngineer, column_step, filter_step, fitted_step
from skyulf.preprocessing.function_steps import FittedFunctionCalculator


def reorder_column(df, params):
    """Reorder the callback's private frame before returning a column."""
    df.sort_values("value", inplace=True)
    result = df["value"] * 10
    return result.to_numpy() if params["array"] else result


def reorder_fitted_column(df, state, params):
    """Exercise the same ordering contract for a fitted apply callback."""
    return reorder_column(df, params)


def learn_empty(df, y, params):
    """Supply valid fitted state without changing training inputs."""
    return {}


def reorder_filter(df):
    """Return a mask ordered by a private sort rather than the original rows."""
    df.sort_values("value", inplace=True)
    return df["value"] >= 20


def drop_filter_row(df):
    """Dropping rows in place must not redefine the required mask length."""
    df.drop(index=df.index[0], inplace=True)
    return df["value"] >= 20


def aligned_after_sort(df):
    """An indexed result may restore the original order after internal sorting."""
    original = df.index.copy()
    df.sort_values("value", inplace=True)
    return (df["value"] * 10).reindex(original)


def aligned_filter_after_sort(df):
    """Internal sorting is safe when the returned mask restores input order."""
    original = df.index.copy()
    df.sort_values("value", inplace=True)
    return (df["value"] >= 20).reindex(original)


def mutate_target(df, y):
    """Learning may edit its private target while the caller retains the original."""
    y[:] = [-5] * len(y)
    return {"mean": float(np.asarray(y).mean())}


def apply_mean(df, state):
    """Use the value learned from the private target."""
    return pd.Series(state["mean"], index=df.index)


def _frame(engine, duplicate=False):
    """Build unsorted rows whose position and label orders differ."""
    frame = pd.DataFrame({"value": [30, 10, 20]}, index=[7, 7, 2] if duplicate else [7, 1, 2])
    return pl.from_pandas(frame) if engine == "polars" else frame


def _pandas(frame):
    """Compare outputs in a common representation without changing row order."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("has_target", [False, True])
def test_column_callback_cannot_redefine_original_row_order(engine, array, fitted, has_target):
    """In-place sorting must never attach computed values to different input rows."""
    frame = _frame(engine)
    if has_target:
        frame = (
            frame.with_columns(pl.lit(1).alias("target"))
            if engine == "polars"
            else frame.assign(target=1)
        )
    original = _pandas(frame).copy()
    step = (
        fitted_step(
            "sorted", learn_empty, reorder_fitted_column, output="derived", params={"array": array}
        )
        if fitted
        else column_step("sorted", reorder_column, output="derived", params={"array": array})
    )
    with pytest.raises(ValueError, match="index or order"):
        FeatureEngineer([step]).fit_transform(frame, target_column="target" if has_target else None)
    assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("callback", [reorder_filter, drop_filter_row])
@pytest.mark.parametrize("duplicate", [False, True])
def test_filter_callback_validates_mask_against_original_rows(engine, callback, duplicate):
    """A mutated callback frame must not make a misaligned mask appear valid."""
    frame = _frame(engine, duplicate)
    original = _pandas(frame).copy()
    target = pd.Series([300, 100, 200])
    step = filter_step("filter", callback, columns=["value"])
    with pytest.raises(ValueError, match="one value per row"):
        FeatureEngineer([step]).fit_transform((frame, target))
    assert_frame_equal(_pandas(frame), original)
    assert target.tolist() == [300, 100, 200]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_callback_can_restore_original_order_after_private_sort(engine):
    """Correctly realigned indexed results remain valid even after internal mutation."""
    frame = _frame(engine)
    step = column_step("aligned", aligned_after_sort, output="derived")
    output, _ = FeatureEngineer([step]).fit_transform(frame)
    assert _pandas(output)["derived"].tolist() == [300, 100, 200]
    step = filter_step("aligned", aligned_filter_after_sort, columns=["value"])
    (kept, target), _ = FeatureEngineer([step]).fit_transform((frame, [300, 100, 200]))
    assert _pandas(kept)["value"].tolist() == [30, 20]
    assert list(target) == [300, 200]


@pytest.mark.parametrize("kind", ["pandas", "polars", "array", "list"])
def test_learn_callback_receives_a_private_writable_target(kind):
    """Callback edits must not change targets used by later nodes or the caller."""
    values = [300, 100, 200]
    targets = {
        "pandas": pd.Series(values, index=[7, 1, 2], name="target"),
        "polars": pl.Series("target", values),
        "array": np.array(values),
        "list": list(values),
    }
    target = targets[kind]
    step = fitted_step("mean", mutate_target, apply_mean, output="derived")
    artifact = FittedFunctionCalculator().fit((_frame("pandas"), target), step["params"])
    assert artifact["state"] == {"mean": -5.0}
    assert list(target) == values


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fitted_callback_preserves_pipeline_target(engine):
    """Training and subsequent transformations retain the caller's target values."""
    frame = _frame(engine)
    target = pd.Series([300, 100, 200], name="target")
    step = fitted_step("mean", mutate_target, apply_mean, output="derived")
    (output, output_target), _ = FeatureEngineer([step]).fit_transform((frame, target))
    assert _pandas(output)["derived"].tolist() == [-5.0, -5.0, -5.0]
    assert list(output_target) == list(target) == [300, 100, 200]
