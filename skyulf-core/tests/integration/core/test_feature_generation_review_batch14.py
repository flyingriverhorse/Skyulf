"""Feature generation must preserve fitted math, unique names and original lag sources."""

import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn import config_context
from sklearn.preprocessing import PowerTransformer

import skyulf
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
    FeatureInteractionApplier,
    FeatureInteractionCalculator,
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.time_series.lag import LagFeaturesApplier, LagFeaturesCalculator
from skyulf.preprocessing.transformations.general import (
    GeneralTransformationApplier,
    GeneralTransformationCalculator,
)


def test_imports_use_selected_source_tree():
    """A snapshot test run must not accidentally validate the unchanged main checkout."""
    expected = os.environ.get("SKYULF_EXPECTED_SOURCE_ROOT")
    if expected is not None:
        assert Path(skyulf.__file__).resolve().parent == Path(expected).resolve() / "skyulf"
    assert Path(skyulf.__file__).is_file()


def _native(frame, engine):
    """Keep native frames while exercising each public engine dispatch path."""
    return pl.from_pandas(frame) if engine == "polars" else frame.copy(deep=True)


def _pandas(frame):
    """Compare public results in a common representation without changing calculations."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("output", ["pandas", "polars"])
def test_fitted_power_replays_under_sklearn_output_config(
    fit_engine, apply_engine, method, standardize, output
):
    """Global sklearn dataframe output must never turn a fitted power rule into a no-op."""
    training = pd.DataFrame({"x": [1.0, 2.0, 4.0, 9.0, 15.0]})
    query = pd.DataFrame({"x": [3.0, 6.0, np.nan, 19.0]}, index=[8, 2, 2, 1])
    oracle = PowerTransformer(method=method, standardize=standardize).fit(training)
    expected = np.asarray(oracle.transform(query)).ravel()
    native = _native(query, apply_engine)
    before = _pandas(native).copy(deep=True)
    with config_context(transform_output=output):
        artifact = GeneralTransformationCalculator().fit(
            _native(training, fit_engine),
            {"transformations": [{"column": "x", "method": method, "standardize": standardize}]},
        )
        result = GeneralTransformationApplier().apply(native, pickle.loads(pickle.dumps(artifact)))
    np.testing.assert_allclose(_pandas(result)["x"], expected, equal_nan=True)
    pd.testing.assert_frame_equal(_pandas(native), before)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ordered_power_fit_uses_prior_transform_with_pandas_output(engine):
    """A later learned rule must fit the actual output of preceding fitted rules."""
    training = pd.DataFrame({"x": [1.0, 2.0, 4.0, 9.0, 15.0]})
    query = pd.DataFrame({"x": [3.0, 6.0, 19.0]})
    first = PowerTransformer(method="yeo-johnson", standardize=False).fit(training)
    second = PowerTransformer(method="yeo-johnson").fit(first.transform(training))
    expected = second.transform(first.transform(query)).ravel()
    with config_context(transform_output="pandas"):
        artifact = GeneralTransformationCalculator().fit(
            _native(training, engine),
            {
                "transformations": [
                    {"column": "x", "method": "yeo-johnson", "standardize": False},
                    {"column": "x", "method": "yeo-johnson"},
                ]
            },
        )
        actual = GeneralTransformationApplier().apply(_native(query, engine), artifact)
    np.testing.assert_allclose(_pandas(actual)["x"], expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("epsilon", [1e-9, 1e-3])
@pytest.mark.parametrize("factor", [-2.0, -1.0, -0.1, 0.0, -0.0, 0.1, 1.0, 2.0])
@pytest.mark.parametrize("constant_only", [False, True])
def test_constant_divisor_preserves_sign_and_zero_policy(engine, epsilon, factor, constant_only):
    """Clamping scalar denominators must match column division, including the epsilon boundary."""
    divisor = epsilon * factor
    frame = _native(pd.DataFrame({"x": [2.0, -2.0], "denominator": [divisor, divisor]}), engine)
    before = _pandas(frame).copy(deep=True)
    config = {
        "epsilon": epsilon,
        "operations": [
            {
                "operation_type": "arithmetic",
                "method": "divide",
                "output_column": "ratio",
                "input_columns": [] if constant_only else ["x"],
                "constants": [2.0, divisor] if constant_only else [divisor],
            }
        ],
    }
    artifact = FeatureGenerationCalculator().fit(frame, config)
    result = FeatureGenerationApplier().apply(frame, pickle.loads(pickle.dumps(artifact)))
    denominator = divisor if abs(divisor) >= epsilon else -epsilon if divisor < 0 else epsilon
    numerator = np.array([2.0, 2.0] if constant_only else [2.0, -2.0])
    np.testing.assert_allclose(_pandas(result)["ratio"], numerator / denominator)
    pd.testing.assert_frame_equal(_pandas(frame), before)


def _collision_case(node, duplicate=False):
    """Describe existing-name and normalized generated-name collisions through public configs."""
    if node == "interaction":
        columns = ["a", "a_x_b", "b_x_c", "c"] if duplicate else ["a", "b"]
        calculator, applier = FeatureInteractionCalculator(), FeatureInteractionApplier()
        colliding = "a_x_b"
    else:
        columns = ["a", "a_b", "b_c", "c"] if duplicate else ["a", "b"]
        calculator, applier = PolynomialFeaturesCalculator(), PolynomialFeaturesApplier()
        colliding = "poly_a_b"
    frame = pd.DataFrame({name: [2.0, 3.0] for name in columns})
    return calculator, applier, frame, {"columns": columns, "degree": 2}, colliding


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["interaction", "polynomial"])
@pytest.mark.parametrize("phase", ["fit", "apply"])
def test_generated_names_cannot_overwrite_or_duplicate_inputs(engine, node, phase):
    """Ambiguous output names must raise consistently before touching any caller column."""
    calculator, applier, training, config, colliding = _collision_case(node)
    artifact = calculator.fit(_native(training, engine), config)
    frame = _native(training.assign(**{colliding: [111.0, 222.0]}), engine)
    before = _pandas(frame).copy(deep=True)
    with pytest.raises(ValueError, match="[Cc]olli|[Uu]nique|[Dd]uplicate"):
        if phase == "fit":
            calculator.fit(frame, config)
        else:
            applier.apply(frame, pickle.loads(pickle.dumps(artifact)))
    pd.testing.assert_frame_equal(_pandas(frame), before)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["interaction", "polynomial"])
def test_distinct_combinations_cannot_share_generated_name(engine, node):
    """Separator normalization cannot silently drop a distinct requested product."""
    calculator, _, frame, config, _ = _collision_case(node, duplicate=True)
    native = _native(frame, engine)
    with pytest.raises(ValueError, match="[Cc]olli|[Uu]nique|[Dd]uplicate"):
        calculator.fit(native, config)
    pd.testing.assert_frame_equal(_pandas(native), _pandas(_native(frame, engine)))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_existing_interaction_bias_retains_established_passthrough(engine):
    """The explicitly skipped existing bias column must retain its original values."""
    frame = _native(pd.DataFrame({"a": [2.0, 3.0], "interaction_bias": [7.0, 8.0]}), engine)
    artifact = FeatureInteractionCalculator().fit(frame, {"columns": ["a"], "include_bias": True})
    actual = FeatureInteractionApplier().apply(frame, artifact)
    pd.testing.assert_frame_equal(_pandas(actual), _pandas(frame))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("sort", [False, True])
def test_lags_read_original_sources_after_generated_name_overwrite(engine, grouped, sort):
    """A generated lag may replace a column but cannot change another lag's source values."""
    frame = pd.DataFrame(
        {
            "x": pd.array([10, 20, 30, 40], dtype="Int64"),
            "x_lag_1": pd.array([2**53 + 1, 2**53 + 3, 2**53 + 5, 2**53 + 7], dtype="Int64"),
            "group": [None, "a", None, "a"],
            "time": [2, 1, 4, 3],
        },
        index=[8, 2, 2, 1],
    )
    target = pd.Series([100, 200, 300, 400], index=frame.index, name="target")
    ordered = frame.sort_values("time", kind="stable") if sort else frame
    original_source = (
        ordered.groupby("group", dropna=False)["x_lag_1"] if grouped else ordered["x_lag_1"]
    )
    expected = original_source.shift(1).tolist()
    config = {
        "columns": ["x", "x_lag_1"],
        "lags": [1, 2],
        "group_by": ["group"] if grouped else None,
        "sort_by": "time" if sort else None,
    }
    native = _native(frame, engine)
    native_target = pl.Series("target", target.tolist()) if engine == "polars" else target
    before = _pandas(native).copy(deep=True)
    artifact = LagFeaturesCalculator().fit((native, native_target), config)
    result, actual_target = LagFeaturesApplier().apply((native, native_target), artifact)
    actual = result["x_lag_1_lag_1"].to_list()
    assert [None if pd.isna(v) else int(v) for v in actual] == [
        None if pd.isna(v) else int(v) for v in expected
    ]
    positions = (
        np.argsort(frame["time"].to_numpy(), kind="stable") if sort else np.arange(len(frame))
    )
    assert actual_target.to_list() == target.iloc[positions].to_list()
    pd.testing.assert_frame_equal(_pandas(native), before)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lags_do_not_discover_newly_generated_columns_as_sources(engine):
    """Configured columns absent from the original frame must remain skipped on both engines."""
    frame = _native(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), engine)
    artifact = LagFeaturesCalculator().fit(frame, {"columns": ["x", "x_lag_1"], "lags": [1]})
    result = LagFeaturesApplier().apply(frame, artifact)
    assert list(result.columns) == ["x", "x_lag_1"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["interaction", "polynomial"])
def test_legacy_artifact_replay_rejects_duplicate_generated_names(engine, node):
    """Existing artifacts must fail clearly rather than overwrite products or return duplicate columns."""
    _, applier, frame, config, _ = _collision_case(node, duplicate=True)
    artifact = dict(config)
    if node == "interaction":
        artifact["combinations"] = [
            list(item) for item in combinations(sorted(config["columns"]), 2)
        ]
    native = _native(frame, engine)
    with pytest.raises(ValueError, match="[Cc]olli|[Uu]nique|[Dd]uplicate"):
        applier.apply(native, pickle.loads(pickle.dumps(artifact)))
    pd.testing.assert_frame_equal(_pandas(native), _pandas(_native(frame, engine)))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_carry_lags_use_original_overlapping_source(engine):
    """Saved context must preserve original feature values when generated lag names overlap."""
    training = pd.DataFrame({"time": [1, 2], "x": [10, 20], "x_lag_1": [100, 200]})
    query = pd.DataFrame({"time": [3, 4], "x": [30, 40], "x_lag_1": [300, 400]})
    artifact = LagFeaturesCalculator().fit(
        _native(training, engine),
        {"columns": ["x", "x_lag_1"], "lags": [1], "sort_by": "time", "history_mode": "carry"},
    )
    actual = LagFeaturesApplier().apply(_native(query, engine), artifact)
    assert actual["x_lag_1_lag_1"].to_list() == [200, 300]
