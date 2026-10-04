"""Preserve threshold defaults, exact values, engine identity, and sampling payloads."""

import copy
import pickle
from contextvars import Context
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.data.dataset import SplitDataset
from skyulf.engines import EngineRegistry, get_engine
from skyulf.engines.pandas_engine import PandasEngine, SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.base import StatefulTransformer
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.preprocessing.transformations.simple import (
    SimpleTransformationApplier,
    SimpleTransformationCalculator,
)


@pytest.mark.parametrize("labels", [(0, 1), ("no", "yes")])
@pytest.mark.parametrize("policy", [None, {}, {"positive_class": 0}])
def test_optional_threshold_policy_matches_default_and_survives_pickle(labels, policy):
    """An explicit null policy must permit the same public tuning lifecycle as omission."""
    negative, positive = labels
    actual_policy = {"positive_class": negative} if policy else policy
    config = {
        "preprocessing": [],
        "modeling": {"type": "logistic_regression", "params": {"C": 0.001}},
        "decision_threshold": actual_policy,
    }
    original_config = copy.deepcopy(config)
    data = pd.DataFrame({"x": range(20), "target": [negative] * 10 + [positive] * 10})
    original_data = data.copy(deep=True)
    pipeline = SkyulfPipeline(config)
    pipeline.fit(data, target_column="target")
    expected = copy.deepcopy(pipeline)
    if actual_policy is None:
        expected.config.pop("decision_threshold")
    holdout = pd.DataFrame({"x": range(6, 14)})
    target = [negative] * 2 + [positive] * 6
    thresholds = pipeline.optimize_thresholds(holdout, target, accuracy_score)
    expected_thresholds = expected.optimize_thresholds(holdout, target, accuracy_score)
    restored = pickle.loads(pickle.dumps(pipeline))
    assert thresholds == expected_thresholds
    np.testing.assert_array_equal(
        restored.predict(holdout, use_tuned_thresholds=True),
        expected.predict(holdout, use_tuned_thresholds=True),
    )
    assert restored.fingerprint() == pipeline.fingerprint()
    assert restored.export_model_card()["fingerprint"] == pipeline.fingerprint()
    pd.testing.assert_frame_equal(data, original_data)
    assert config == original_config


@pytest.mark.parametrize("policy", [False, 0, [], "invalid"])
def test_non_mapping_threshold_policy_is_not_silently_treated_as_absent(policy):
    """Accepting null cannot silently disable other malformed policy values."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": "logistic_regression"},
            "decision_threshold": policy,
        }
    )
    pipeline.fit(pd.DataFrame({"x": range(8), "target": [0] * 4 + [1] * 4}), "target")
    with pytest.raises((AttributeError, TypeError, ValueError)):
        pipeline.optimize_thresholds(
            pd.DataFrame({"x": [2, 3, 4, 5]}), [0, 0, 1, 1], accuracy_score
        )


@pytest.mark.parametrize("method", ["with_column", "setitem"])
@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("same_index", [False, True])
def test_positional_assignment_preserves_nullable_integer_values(
    method, dtype, missing, same_index
):
    """Positional assignment must preserve exact wide integers and nullable storage."""
    frame = pd.DataFrame({"x": [1, 2, 3]}, index=[7, 7, 9])
    original_frame = frame.copy(deep=True)
    values = pd.Series(
        [2**53 + 1, pd.NA if missing else 5, 2**53 + 3],
        dtype=dtype,
        index=frame.index if same_index else [30, 20, 10],
        name="source",
    )
    original_values = values.copy(deep=True)
    wrapper = SkyulfPandasWrapper(frame)
    if method == "with_column":
        output = wrapper.with_column("exact", values).to_native()
        pd.testing.assert_frame_equal(frame, original_frame)
    else:
        wrapper["exact"] = values
        output = wrapper.to_native()
    expected = pd.Series(original_values.array, index=frame.index, name="exact")
    pd.testing.assert_series_equal(output["exact"], expected)
    pd.testing.assert_series_equal(values, original_values)
    assert output["exact"].iloc[0] == 2**53 + 1


class CustomFrame(pd.DataFrame):
    """Represent a real pandas subclass defined outside the pandas package."""

    @property
    def _constructor(self):
        """Retain the public subclass through pandas copy and assignment operations."""
        return CustomFrame


@pytest.mark.parametrize("fallback", ["pandas", "polars"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_pandas_subclass_public_dispatch_preserves_rows(fallback, wrapped, subclass):
    """A recognized frame's ancestors must override unrelated active-engine defaults."""
    frame_type = CustomFrame if subclass else pd.DataFrame
    frame = frame_type({"x": [1.0, 4.0, 9.0], "keep": [3, 2, 1]}, index=[9, 9, 2])
    original = frame.copy(deep=True)
    payload = SkyulfPandasWrapper(frame) if wrapped else frame
    config = {"transformations": [{"column": "x", "method": "sqrt"}]}

    def apply_in_context():
        """Exercise actual dispatch with an isolated caller-local fallback engine."""
        EngineRegistry.set_active_engine(fallback)
        assert get_engine(payload) is PandasEngine
        assert isinstance(EngineRegistry.wrap(payload), SkyulfPandasWrapper)
        params = dict(SimpleTransformationCalculator().fit(payload, config))
        return SimpleTransformationApplier().apply(payload, params)

    output = Context().run(apply_in_context)
    expected = original.copy()
    expected["x"] = [1.0, 2.0, 3.0]
    pd.testing.assert_frame_equal(output, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert output.index.tolist() == [9, 9, 2]


@pytest.fixture
def sampler_components():
    """Load optional samplers only for tests that exercise resampling behavior."""
    pytest.importorskip("imblearn")
    from skyulf.preprocessing.resampling import (
        OversamplingApplier,
        OversamplingCalculator,
        UndersamplingApplier,
        UndersamplingCalculator,
    )

    return {
        "random_over": (OversamplingCalculator, OversamplingApplier),
        "smote": (OversamplingCalculator, OversamplingApplier),
        "random_under_sampling": (UndersamplingCalculator, UndersamplingApplier),
    }


def _sampling_input(engine, wrapped, shape):
    """Build equivalent inputs with a named target and non-unique pandas row labels."""
    frame: Any = pd.DataFrame(
        {"label": [2, 2, 2, 2, 3, 3], "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
        index=[9, 9, 3, 3, 1, 1],
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    target = frame["label"]
    if shape == "pair":
        frame = frame.drop("label") if engine == "polars" else frame.drop(columns=["label"])
    if wrapped:
        frame = SkyulfPolarsWrapper(frame) if engine == "polars" else SkyulfPandasWrapper(frame)
    if shape == "pair":
        return frame, target
    if shape == "placeholder":
        return frame, None
    return frame


def _native_frame(payload):
    """Read supported wrapper frames without assuming their private implementation."""
    return payload.to_native() if hasattr(payload, "to_native") else payload


def _as_pandas(payload):
    """Compare engine-native values and column names without casting numerical values."""
    native = _native_frame(payload)
    return native.to_pandas() if isinstance(native, pl.DataFrame) else native


@pytest.mark.parametrize("method", ["random_over", "random_under_sampling", "smote"])
@pytest.mark.parametrize("shape", ["embedded", "pair", "placeholder"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_weighted_resampling_preserves_public_payload_contract(
    sampler_components, method, shape, engine, wrapped
):
    """Weights must change neither payload shape nor sampled rows from unweighted execution."""
    payload = _sampling_input(engine, wrapped, shape)
    source = payload[0] if isinstance(payload, tuple) else payload
    original = _as_pandas(source).copy(deep=True)
    original_target = copy.deepcopy(payload[1]) if isinstance(payload, tuple) else None
    calculator, applier = sampler_components[method]
    config = {
        "method": method,
        "target_column": "label",
        "random_state": 19,
        "k_neighbors": 1,
        "synthetic_weight": "class_mean",
    }
    original_config = copy.deepcopy(config)
    weights = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    expected, no_weights = StatefulTransformer(
        calculator(), applier(), node_id="sampling"
    ).fit_transform_weighted(payload, config, None)
    actual, actual_weights = StatefulTransformer(
        calculator(), applier(), node_id="sampling"
    ).fit_transform_weighted(payload, config, weights)
    assert no_weights is None
    assert type(actual) is type(expected)
    if isinstance(expected, tuple):
        actual_X, actual_y = actual
        expected_X, expected_y = expected
        np.testing.assert_array_equal(np.asarray(actual_y), np.asarray(expected_y))
        assert actual_y.name == expected_y.name == "label"
    else:
        actual_X, expected_X = actual, expected
    assert type(actual_X) is type(expected_X)
    pd.testing.assert_frame_equal(_as_pandas(actual_X), _as_pandas(expected_X))
    expected_weights = (
        [10, 20, 30, 40, 50, 60, 55, 55]
        if method == "smote"
        else (np.asarray(_native_frame(actual_X)["x"]) + 1) * 10
    )
    np.testing.assert_array_equal(actual_weights, expected_weights)
    np.testing.assert_array_equal(weights, [10, 20, 30, 40, 50, 60])
    if original_target is not None:
        np.testing.assert_array_equal(np.asarray(payload[1]), np.asarray(original_target))
    pd.testing.assert_frame_equal(_as_pandas(source), original)
    assert config == original_config


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_weighted_feature_engineering_keeps_target_available_to_next_column_step(
    sampler_components, engine, wrapped
):
    """A weighted sampler cannot hide an embedded target from a configured column consumer."""
    payload = _sampling_input(engine, wrapped, "embedded")
    engineer = FeatureEngineer(
        [
            {"name": "sample", "transformer": "Oversampling", "params": {"method": "random_over"}},
            {
                "name": "square_target",
                "transformer": "SimpleTransformation",
                "params": {"transformations": [{"column": "label", "method": "square"}]},
            },
        ]
    )
    actual, _ = engineer.fit_transform(
        payload, target_column="label", sample_weight=[10, 20, 30, 40, 50, 60]
    )
    native = _native_frame(actual)
    assert not isinstance(actual, tuple)
    np.testing.assert_array_equal(
        np.asarray(native["label"]), np.where(np.asarray(native["x"]) < 4, 4, 9)
    )
    np.testing.assert_array_equal(engineer.train_sample_weight_, (np.asarray(native["x"]) + 1) * 10)
    assert len(native) == 8


@pytest.mark.parametrize("shape", ["embedded", "pair"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_weighted_split_sampling_keeps_heldout_slots_and_training_shape(
    sampler_components, shape, engine, wrapped
):
    """Training-only sampling must preserve held-out objects and transport positional weights."""
    train = _sampling_input(engine, wrapped, shape)
    heldout = _sampling_input(engine, wrapped, shape)
    weights = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    dataset = SplitDataset(
        train=train, test=heldout, validation=heldout, train_sample_weight=weights
    )
    calculator, applier = sampler_components["random_over"]
    transformer = StatefulTransformer(
        calculator(), applier(), node_id="sampling", apply_on_test=False, apply_on_validation=False
    )
    result = transformer.fit_transform(
        dataset, {"method": "random_over", "target_column": "label", "random_state": 19}
    )
    assert isinstance(result, SplitDataset)
    assert isinstance(result.train, tuple) == (shape == "pair")
    sampled = result.train[0] if isinstance(result.train, tuple) else result.train
    np.testing.assert_array_equal(
        result.train_sample_weight, (np.asarray(_native_frame(sampled)["x"]) + 1) * 10
    )
    assert result.test is result.validation is heldout
    assert dataset.train is train
    np.testing.assert_array_equal(weights, [10, 20, 30, 40, 50, 60])
