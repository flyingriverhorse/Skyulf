"""Temporal inference must reject sorted outputs without input-row provenance."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines import EngineRegistry
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.fixture(params=[("pandas", False), ("pandas", True), ("polars", False), ("polars", True)])
def frame_kind(request):
    """Exercise native and wrapped frames on both supported engines."""
    return request.param


def _frame(frame_kind, *, sorted_rows: bool = False) -> Any:
    """Include tied sort keys, duplicate indices, and a null group key."""
    frame = pd.DataFrame(
        {"time": [2, 1, 1, 3], "value": [20.0, 10.0, 11.0, 30.0], "group": ["a", "a", "b", None]},
        index=[7, 7, 2, 2],
    )
    if sorted_rows:
        frame = frame.iloc[[1, 2, 0, 3]]
    engine_name, wrapped = frame_kind
    native = pl.from_pandas(frame) if engine_name == "polars" else frame
    return EngineRegistry.wrap(native) if wrapped else native


def _step(transformer: str, **overrides) -> dict[str, Any]:
    """Build an effective temporal step whose sort can change row order."""
    params: dict[str, Any] = {"columns": ["value"], "sort_by": "time", "group_by": ["group"]}
    params.update(
        {"lags": [1]} if transformer == "LagFeatures" else {"window": 2, "min_periods": 1}
    )
    params.update(overrides)
    return {"name": "temporal", "transformer": transformer, "params": params}


def _as_pandas(frame) -> pd.DataFrame:
    """Expose values without relying on one engine's indexing interface."""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


@pytest.mark.parametrize("transformer", ["LagFeatures", "RollingAggregate"])
def test_prediction_transform_rejects_temporal_reordering(frame_kind, transformer):
    """Equal row counts must not conceal a permutation of the submitted observations."""
    frame = _frame(frame_kind)
    engineer = FeatureEngineer([_step(transformer)])
    engineer.fit_transform(frame)
    ordinary = engineer.transform(frame)
    assert _as_pandas(ordinary)["value"].tolist() == [10.0, 11.0, 20.0, 30.0]
    with pytest.raises(ValueError, match="temporal.*row order.*Sort.*time.*prediction"):
        engineer.transform(frame, preserve_rows=True)
    assert _as_pandas(frame)["value"].tolist() == [20.0, 10.0, 11.0, 30.0]


@pytest.mark.parametrize("transformer", ["LagFeatures", "RollingAggregate"])
def test_sorted_temporal_prediction_preserves_values_and_calls_applier_once(
    frame_kind, transformer, monkeypatch
):
    """Stable ties and duplicate indices are safe when temporal sorting leaves positions intact."""
    frame = _frame(frame_kind, sorted_rows=True)
    engineer = FeatureEngineer([_step(transformer)])
    engineer.fit_transform(frame)
    applier = engineer.fitted_steps[0]["applier"]
    original_apply = applier.apply
    calls = []

    def recording_apply(data, artifact):
        """Count real executions while retaining the actual temporal calculation."""
        calls.append(1)
        return original_apply(data, artifact)

    monkeypatch.setattr(applier, "apply", recording_apply)
    result = _as_pandas(engineer.transform(frame, preserve_rows=True))
    assert result["value"].tolist() == [10.0, 11.0, 20.0, 30.0]
    assert len(result.columns) == 4
    assert calls == [1]


@pytest.mark.parametrize("with_target", [False, True])
def test_temporal_position_ids_never_reach_a_later_target_encoder(frame_kind, with_target):
    """A local order probe must not encode row identifiers or fabricate target outputs."""
    frame = _frame(frame_kind, sorted_rows=True)
    labels = np.array(["yes", "no", "yes", "no"])
    engineer = FeatureEngineer(
        [_step("RollingAggregate"), {"name": "labels", "transformer": "LabelEncoder", "params": {}}]
    )
    engineer.fit_transform((frame, labels))
    data = (frame, labels) if with_target else frame
    result = engineer.transform(data, preserve_rows=True)
    if with_target:
        result, target = result
        np.testing.assert_array_equal(np.asarray(target), [1, 0, 1, 0])
    else:
        assert not isinstance(result, tuple)
    assert _as_pandas(result)["value"].tolist() == [10.0, 11.0, 20.0, 30.0]
    assert list(_as_pandas(result).columns) == ["time", "value", "group", "value_roll_mean_2"]


def test_temporal_count_error_precedes_order_error(frame_kind):
    """Filtering must identify lost samples even when the same temporal step also sorts."""
    frame = _frame(frame_kind)
    engineer = FeatureEngineer([_step("LagFeatures", drop_na=True)])
    engineer.fit_transform(frame)
    with pytest.raises(ValueError, match="temporal.*row count.*4.*1"):
        engineer.transform(frame, preserve_rows=True)
    assert len(frame) == 4


def test_ordinary_rolling_then_lag_keeps_sorted_intermediate_rows(frame_kind):
    """Restoring order between steps would change the later lag's feature values."""
    frame = _frame(frame_kind)
    engineer = FeatureEngineer(
        [
            _step("RollingAggregate", group_by=None),
            {
                "name": "lag",
                "transformer": "LagFeatures",
                "params": {"columns": ["value_roll_mean_2"], "lags": [1]},
            },
        ]
    )
    engineer.fit_transform(frame)
    result = _as_pandas(engineer.transform(frame))
    assert result["time"].tolist() == [1, 1, 2, 3]
    np.testing.assert_allclose(result["value_roll_mean_2"], [10, 10.5, 15.5, 25])
    np.testing.assert_allclose(result["value_roll_mean_2_lag_1"], [np.nan, 10, 10.5, 15.5])


@pytest.mark.parametrize("transformer", ["LagFeatures", "RollingAggregate"])
def test_group_only_temporal_prediction_keeps_input_positions(frame_kind, transformer):
    """Grouping must not be rejected when it computes features without sorting observations."""
    frame = _frame(frame_kind)
    engineer = FeatureEngineer([_step(transformer, sort_by=None)])
    engineer.fit_transform(frame)
    result = _as_pandas(engineer.transform(frame, preserve_rows=True))
    assert result["value"].tolist() == [20.0, 10.0, 11.0, 30.0]
