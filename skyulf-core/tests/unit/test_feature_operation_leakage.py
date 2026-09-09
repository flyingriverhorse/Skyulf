"""Regression checks for fitted and row-local feature operation boundaries."""

import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation.generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)
from skyulf.preprocessing.feature_generation.polynomial import (
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.time_series.lag import LagFeaturesApplier, LagFeaturesCalculator
from skyulf.preprocessing.time_series.rolling import (
    RollingAggregateApplier,
    RollingAggregateCalculator,
)
from skyulf.preprocessing.transformations.general import (
    GeneralTransformationApplier,
    GeneralTransformationCalculator,
)


def _frame(data: dict[str, list[Any]], engine: str) -> Any:
    """Build equivalent pandas and Polars inputs for the public node boundary."""
    frame = pd.DataFrame(data)
    return frame if engine == "pandas" else pl.from_pandas(frame)


def _values(frame: Any, column: str) -> list[Any]:
    """Read output values without hiding which engine the applier returned."""
    return frame[column].to_list()


def _group_operation(method: str = "mean") -> dict[str, Any]:
    """Describe a categorical-key aggregate with a stable output name."""
    return {
        "operation_type": "group_agg",
        "method": method,
        "input_columns": ["group"],
        "secondary_columns": ["value"],
        "output_column": "group_value",
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mean", 20.0),
        ("sum", 40.0),
        ("count", 2.0),
        ("min", 10.0),
        ("max", 30.0),
        ("std", 14.142135623730951),
        ("median", 20.0),
    ],
)
def test_group_aggregates_use_training_values(engine: str, method: str, expected: float) -> None:
    """Held-out magnitudes and batch sizes must never become fitted group statistics."""
    train = _frame({"group": ["A", "A", "A"], "value": [10.0, 30.0, None]}, engine)
    config = {"operations": [_group_operation(method)]}
    params = FeatureGenerationCalculator().fit(train, config)
    saved = copy.deepcopy(params)
    heldout = _frame({"group": ["A", "A"], "value": [1000.0, -2000.0]}, engine)
    singleton = _frame({"group": ["A"], "value": [1000.0]}, engine)

    batch_out = FeatureGenerationApplier().apply(heldout, params)
    single_out = FeatureGenerationApplier().apply(singleton, params)

    assert _values(batch_out, "group_value") == pytest.approx([expected, expected])
    assert _values(single_out, "group_value") == pytest.approx([expected])
    assert params == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregate_inference_needs_only_group_key(engine: str) -> None:
    """Inference must not require the training-only aggregation source column."""
    train = _frame({"group": ["A", "A"], "value": [10.0, 30.0]}, engine)
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    heldout = _frame({"group": ["A", "B"]}, engine)

    out = FeatureGenerationApplier().apply(heldout, params)

    assert "group_value" in out.columns
    assert _values(out, "group_value")[0] == 20.0
    assert pd.isna(_values(out, "group_value")[1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregate_reuses_null_group_and_preserves_unseen_missing(engine: str) -> None:
    """Missing keys learn a training group while unseen keys never use test values."""
    train = _frame({"group": [None, None, "A"], "value": [4.0, 8.0, 20.0]}, engine)
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    heldout = _frame({"group": [None, "A", "new"], "value": [100.0, 200.0, 300.0]}, engine)

    out = FeatureGenerationApplier().apply(heldout, params)

    assert _values(out, "group_value")[:2] == [6.0, 20.0]
    assert pd.isna(_values(out, "group_value")[2])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregate_all_missing_training_values_stay_missing(engine: str) -> None:
    """An empty learned mean must not trigger an inference-time aggregate fallback."""
    train = _frame({"group": ["A", "A"], "value": [float("nan"), float("nan")]}, engine)
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    heldout = _frame({"group": ["A", "new"], "value": [100.0, 200.0]}, engine)

    out = FeatureGenerationApplier().apply(heldout, params)

    assert all(pd.isna(value) for value in _values(out, "group_value"))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregate_fit_respects_chained_operations(engine: str) -> None:
    """Aggregates must learn earlier generated training columns with stable operation names."""
    train = _frame({"group": ["A", "A"], "value": [10.0, 30.0]}, engine)
    config = {
        "operations": [
            {
                "operation_type": "arithmetic",
                "method": "multiply",
                "input_columns": ["value"],
                "constants": [2],
                "output_column": "doubled",
            },
            {
                "operation_type": "group_agg",
                "input_columns": ["group"],
                "secondary_columns": ["doubled"],
            },
            {
                "operation_type": "ratio",
                "input_columns": ["value"],
                "secondary_columns": ["group_agg_1"],
                "output_column": "relative",
            },
        ]
    }
    params = FeatureGenerationCalculator().fit(train, config)
    heldout = _frame({"group": ["A"], "value": [100.0]}, engine)

    out = FeatureGenerationApplier().apply(heldout, params)

    assert _values(out, "doubled") == [200.0]
    assert _values(out, "group_agg_1") == [40.0]
    assert _values(out, "relative") == [2.5]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregate_config_mutation_cannot_change_fitted_operation(engine: str) -> None:
    """A caller editing its config must not silently change an already fitted feature."""
    train = _frame({"group": ["A", "A"], "value": [10.0, 30.0]}, engine)
    config = {"operations": [_group_operation()]}
    params = FeatureGenerationCalculator().fit(train, config)
    config["operations"][0]["method"] = "sum"

    out = FeatureGenerationApplier().apply(train, params)

    assert _values(out, "group_value") == [20.0, 20.0]


def test_public_group_aggregate_rejects_unfitted_artifact() -> None:
    """A legacy unfitted artifact must request refitting instead of learning from predictions."""
    frame = pd.DataFrame({"group": ["A", "A"], "value": [100.0, 200.0]})

    with pytest.raises(ValueError, match="(?i)fit"):
        FeatureGenerationApplier().apply(frame, {"operations": [_group_operation()]})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregate_empty_training_map_stays_empty(engine: str) -> None:
    """Zero observed training groups must never allow a held-out fallback fit."""
    train = _frame({"group": [], "value": []}, engine)
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    heldout = _frame({"group": ["new"], "value": [100.0]}, engine)

    out = FeatureGenerationApplier().apply(heldout, params)

    assert "group_value" in out.columns
    assert pd.isna(_values(out, "group_value")[0])


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_group_aggregate_numeric_keys_survive_serialization(
    fit_engine: str, apply_engine: str
) -> None:
    """Persisted group lookups must retain numeric key types across engine changes."""
    train = _frame({"group": [1, 1, 2], "value": [10.0, 30.0, 7.0]}, fit_engine)
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    restored = json.loads(json.dumps(params))
    heldout = _frame({"group": [2, 1, 3]}, apply_engine)

    out = FeatureGenerationApplier().apply(heldout, restored)

    assert _values(out, "group_value")[:2] == [7.0, 20.0]
    assert pd.isna(_values(out, "group_value")[2])


def test_group_aggregate_categorical_null_keys_and_duplicate_indices() -> None:
    """Categorical missing groups and duplicate row labels must preserve row alignment."""
    train = pd.DataFrame(
        {"group": pd.Categorical(["A", None, None]), "value": [10.0, 4.0, 8.0]},
        index=[5, 5, 1],
    )
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    heldout = pd.DataFrame({"group": pd.Categorical([None, "A", "new"])}, index=[9, 9, 3])

    out = FeatureGenerationApplier().apply(heldout, params)

    assert list(out.index) == [9, 9, 3]
    assert out["group_value"].tolist()[:2] == [6.0, 10.0]
    assert pd.isna(out["group_value"].iloc[2])


def test_group_aggregate_polars_nan_keys_reuse_training_null_group() -> None:
    """Polars IEEE NaN keys must use the same learned missing group as pandas."""
    train = pl.DataFrame({"group": [float("nan"), float("nan"), 1.0], "value": [4.0, 8.0, 20.0]})
    params = FeatureGenerationCalculator().fit(train, {"operations": [_group_operation()]})
    heldout = pl.DataFrame({"group": [float("nan"), 1.0]})

    out = FeatureGenerationApplier().apply(heldout, params)

    assert out["group_value"].to_list() == [6.0, 20.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_fitted_general_power_transform_is_batch_independent(engine: str, method: str) -> None:
    """Saved lambda and scaling statistics must survive extreme held-out companions."""
    train = _frame({"value": [1.0, 2.0, 4.0, 10.0, 30.0]}, engine)
    config = {"transformations": [{"column": "value", "method": method}]}
    params = GeneralTransformationCalculator().fit(train, config)
    heldout = _frame({"value": [3.0, 100000.0]}, engine)
    singleton = _frame({"value": [3.0]}, engine)

    batch_out = GeneralTransformationApplier().apply(heldout, params)
    single_out = GeneralTransformationApplier().apply(singleton, params)

    assert np.isfinite(_values(batch_out, "value")).all()
    assert _values(batch_out, "value")[0] == pytest.approx(_values(single_out, "value")[0])
    assert _values(batch_out, "value")[0] != 3.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_polynomial_refit_does_not_learn_prediction_statistics(engine: str) -> None:
    """Sklearn's polynomial fit may rebuild the basis without using other rows' values."""
    train = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, engine)
    params = PolynomialFeaturesCalculator().fit(train, {"columns": ["a", "b"], "degree": 2})
    heldout = _frame({"a": [3.0, 1000.0], "b": [4.0, -1000.0]}, engine)

    out = PolynomialFeaturesApplier().apply(heldout, params)

    assert _values(out, "poly_a_pow_2")[0] == 9.0
    assert _values(out, "poly_a_b")[0] == 12.0
    assert _values(out, "poly_b_pow_2")[0] == 16.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lag_and_rolling_respect_sorted_past_within_groups(engine: str) -> None:
    """Temporal features may use their current batch's past but must not reach future rows."""
    frame = _frame(
        {"time": [3, 1, 2, 2], "group": ["A", "A", "A", "B"], "value": [900.0, 10.0, 20.0, 500.0]},
        engine,
    )
    config = {"columns": ["value"], "sort_by": "time", "group_by": ["group"]}
    lag_params = LagFeaturesCalculator().fit(frame, {**config, "lags": [1]})
    rolling_params = RollingAggregateCalculator().fit(frame, {**config, "window": 2})

    lagged = LagFeaturesApplier().apply(frame, lag_params)
    rolled = RollingAggregateApplier().apply(frame, rolling_params)

    assert _values(lagged, "value_lag_1")[1] == 10.0
    assert _values(rolled, "value_roll_mean_2")[:2] == [10.0, 15.0]
