"""Merged fold branches must preserve observations used by positional joins."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter, MergedBranchFoldAdapter
from skyulf.preprocessing.pipeline import FeatureEngineer


def _step(transformer, **params):
    """Create an unfitted step without sharing mutable test configuration."""
    return {"name": transformer, "transformer": transformer, "params": params}


def _payload(engine):
    """Use unsorted observations and distinct labels to expose positional mixing."""
    X = pd.DataFrame({"time": [3, 1, 2], "value": [30.0, 10.0, 20.0]}, index=[8, 8, 4])
    y = np.array([300, 100, 200])
    return (pl.from_pandas(X) if engine == "polars" else X), y


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["first_wins", "last_wins"])
@pytest.mark.parametrize(
    "branch",
    [
        _step("LagFeatures", columns=["value"], sort_by="time"),
        _step("RollingAggregate", columns=["value"], sort_by="time", window=2),
        _step("LagFeatures", columns=["value"], drop_na=True),
        _step("ManualBounds", bounds={"value": {"lower": 15}}),
    ],
    ids=["sorted-lag", "sorted-rolling", "filtered-lag", "bounds"],
)
def test_merged_folds_reject_branches_that_change_observation_order_or_count(
    engine, strategy, branch
):
    """Unsupported branch shapes must fail before returning mispaired training rows."""
    X, y = _payload(engine)
    with pytest.raises(ValueError, match="row counts|row order"):
        adapter = MergedBranchFoldAdapter(
            [[branch], [_step("StandardScaler", columns=["value"])]], strategy, "target"
        )
        adapter.fit_transform(X, y)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("transformer", ["LagFeatures", "RollingAggregate"])
def test_unsorted_time_series_branches_keep_observation_identity(engine, transformer):
    """Time-series features remain usable when sorting and row filtering are disabled."""
    X, y = _payload(engine)
    adapter = MergedBranchFoldAdapter(
        [[_step(transformer, columns=["value"], window=2)], [_step("StandardScaler")]],
        "first_wins",
        "target",
    )
    frame, target = adapter.fit_transform(X, y)
    assert frame["time"].tolist() == [3, 1, 2]
    np.testing.assert_array_equal(target, y)
    feature = "value_lag_1" if transformer == "LagFeatures" else "value_roll_mean_2"
    expected = [np.nan, 30, 10] if transformer == "LagFeatures" else [30, 20, 15]
    np.testing.assert_allclose(frame[feature], expected, equal_nan=True)
    applied, applied_y = adapter.transform(X, y)
    pd.testing.assert_frame_equal(applied, frame)
    np.testing.assert_array_equal(applied_y, y)
    inference, missing_y = adapter.transform(X, None)
    pd.testing.assert_frame_equal(inference, frame)
    assert missing_y is None


@pytest.mark.parametrize(
    "step, expected",
    [
        (_step("ManualBounds", bounds={"value": {"lower": 15}}), True),
        (_step("LagFeatures", columns=["value"], drop_na=True), True),
        (_step("LagFeatures", columns=["value"], drop_na=False), False),
    ],
)
def test_single_branch_row_count_flag_accounts_for_configured_filtering(step, expected):
    """The single-branch adapter must describe the filtering it actually performs."""
    adapter = FeatureEngineerFoldAdapter([step], "target")
    X, y = _payload("pandas")
    frame, target = adapter.fit_transform(X, y)
    assert len(frame) == len(target) == (2 if expected else 3)
    assert adapter.changes_row_count is expected


def test_merged_configuration_is_isolated_after_validation():
    """Caller mutations cannot introduce unsafe sorting after branch validation."""
    branches = [[_step("RollingAggregate", columns=["value"], window=2)]]
    adapter = MergedBranchFoldAdapter(branches, "last_wins", "target")
    branches[0][0]["params"]["sort_by"] = "time"
    X, y = _payload("pandas")
    frame, target = adapter.fit_transform(X, y)
    assert frame["time"].tolist() == [3, 1, 2]
    np.testing.assert_array_equal(target, y)


@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_merged_adapter_rejects_mismatched_target_count(stage):
    """A positional merge must not publish a target with a different number of rows."""
    X, y = _payload("pandas")
    adapter = MergedBranchFoldAdapter([[_step("StandardScaler")]], "last_wins", "target")
    adapter.fit_transform(X, y)
    apply = adapter.fit_transform if stage == "fit" else adapter.transform
    with pytest.raises(ValueError, match="row counts"):
        apply(X, y[:2])


@pytest.mark.parametrize("stage", ["fit", "transform"])
@pytest.mark.parametrize("changed", ["features", "target"])
def test_branch_output_contract_failure_preserves_the_last_fitted_adapter(
    monkeypatch, stage, changed
):
    """A misbehaving branch must neither produce a mismatched merge nor replace fitted state."""
    X, y = _payload("pandas")
    adapter = MergedBranchFoldAdapter([[_step("StandardScaler")]], "last_wins", "target")
    adapter.fit_transform(X, y)
    fitted = adapter._engineers

    def invalid_output(self, payload):
        """Model a branch that violates its advertised row-preserving contract."""
        frame, labels = payload
        output = (frame.iloc[:-1], labels) if changed == "features" else (frame, labels[:-1])
        return (output, {}) if stage == "fit" else output

    method = "fit_transform" if stage == "fit" else "transform"
    monkeypatch.setattr(FeatureEngineer, method, invalid_output)
    with pytest.raises(ValueError, match="branch changed row counts"):
        getattr(adapter, method)(X, y)
    assert adapter._engineers is fitted
