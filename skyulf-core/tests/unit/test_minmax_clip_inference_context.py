"""Existing affine and fixed-bound appliers expose their reviewed local Polars context."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.partition_safety import _inspect_step, require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64"])
@pytest.mark.parametrize(
    "node,config",
    [
        ("MinMaxScaler", {"columns": ["x"], "feature_range": [-2, 7]}),
        ("ClipValues", {"bounds": {"x": {"lower": 1, "upper": 7}}}),
        ("ClipValues", {"bounds": {"x": {"lower": 0.5, "upper": 7.5}}}),
    ],
)
def test_local_context_reuses_fitted_bounds_and_affine_state(
    engine, dtype, node, config, monkeypatch
):
    """Local declarations must describe unchanged saved arithmetic across request boundaries."""
    training = pd.DataFrame({"x": pd.Series([0, 1, 2, 3, 10], dtype=dtype), "keep": range(5)})
    floating = dtype.startswith("float")
    values = [None, np.nan, np.inf, -0.0, 6.0, 100.0] if floating else [0, 6, 100]
    sample = pd.DataFrame({"x": pd.Series(values, dtype=dtype), "keep": range(len(values))})
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    calculator = NodeRegistry.get_calculator(node)
    state = calculator().fit(training, config)
    saved = pickle.dumps(state)
    before = pickle.dumps(sample)

    def forbidden(*args, **kwargs):
        """Replaying saved parameters must never learn statistics from scoring rows."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(calculator, "fit", forbidden)
    capability = get_inference_capability(node, config, state, engine=engine)
    assert capability is not None
    assert (capability.context, capability.row_effect, capability.execution_kind) == (
        "row",
        "preserve",
        "local" if engine == "polars" else "python_batch",
    )
    applier = NodeRegistry.get_applier(node)()
    full = applier.apply(sample, state)
    if isinstance(sample, pl.DataFrame):
        pieces = [
            applier.apply(sample.slice(i, 1).lazy(), state).collect() for i in range(len(sample))
        ]
        assert_frame_equal(pl.concat(pieces), full, check_exact=True)
        assert_frame_equal(applier.apply(sample.lazy(), state).collect(), full, check_exact=True)
        assert_frame_equal(applier.apply(sample.reverse(), state), full.reverse(), check_exact=True)
        assert_frame_equal(applier.apply(sample.head(0).lazy(), state).collect(), full.head(0))
    else:
        pieces = [applier.apply(sample.iloc[i : i + 1], state) for i in range(len(sample))]
        pd.testing.assert_frame_equal(pd.concat(pieces), full, check_exact=True)
        pd.testing.assert_frame_equal(
            applier.apply(sample.iloc[::-1], state), full.iloc[::-1], check_exact=True
        )
        pd.testing.assert_frame_equal(applier.apply(sample.head(0), state), full.head(0))
    assert pickle.dumps(sample) == before
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize(
    "node,config,bad_config",
    [
        ("MinMaxScaler", {"columns": ["x"]}, {"columns": ["x"], "feature_range": [-1, 1]}),
        ("ClipValues", {"bounds": {"x": {"lower": 0}}}, {"bounds": {"x": {"lower": 1}}}),
    ],
)
def test_local_metadata_keeps_validation_and_worker_boundaries(
    node, config, bad_config, monkeypatch
):
    """Adding local metadata must retain pure inspection, abstention and execution-kind admission."""
    calculator = NodeRegistry.get_calculator(node)
    applier = NodeRegistry.get_applier(node)
    state = calculator().fit(pl.DataFrame({"x": [0.0, 1.0, 2.0]}), config)
    saved = pickle.dumps(state)

    def forbidden(*args, **kwargs):
        """Metadata inspection must not invoke a transform or fit."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(calculator, "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, config, state, engine="polars") is not None
    assert get_inference_capability(node, bad_config, state, engine="polars") is None
    assert (
        get_inference_capability(node, config, {**state, "type": "invalid"}, engine="polars")
        is None
    )
    overridden = applier()
    overridden.apply = forbidden
    assert (
        get_inference_capability(node, config, state, engine="polars", applier=overridden) is None
    )
    for kind in ("python_batch", "native"):
        with pytest.raises(UnsupportedExecutionError):
            require_capability(node, "apply", "polars", config=config, execution_kind=kind)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_minmax_explicit_empty_selection_is_a_saved_identity(engine):
    """An intentionally disabled scaler must keep exact values, dtypes and empty row shape."""
    frame = pd.DataFrame({"x": pd.Series([1, None, 3], dtype="Int64")})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": [], "feature_range": (-2, 7)}
    state = NodeRegistry.get_calculator("MinMaxScaler")().fit(frame, config)
    assert state == {}
    state = pickle.loads(pickle.dumps(state))
    original = deepcopy(frame)
    before = pickle.dumps((config, state))
    capability = get_inference_capability("MinMaxScaler", config, state, engine=engine)
    assert capability is not None and capability.context == "row"
    applier = NodeRegistry.get_applier("MinMaxScaler")()
    for positions in ([0, 1, 2], [1], [2, 0], []):
        sample = frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]
        actual = applier.apply(sample, state)
        if isinstance(sample, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, sample, check_exact=True)
        else:
            assert_frame_equal(actual, sample, check_exact=True)
    record = {
        "name": "empty",
        "type": "MinMaxScaler",
        "artifact": state,
        "params": config,
        "applier": applier,
    }
    recipe = {"name": "empty", "transformer": "MinMaxScaler", "params": config}
    assert _inspect_step(record, recipe).action == "apply"
    if isinstance(frame, pd.DataFrame):
        assert isinstance(original, pd.DataFrame)
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
    else:
        assert isinstance(original, pl.DataFrame)
        assert_frame_equal(frame, original, check_exact=True)
    assert pickle.dumps((config, state)) == before


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"columns": ["x"]},
        {"columns": [], "_auto_columns": True},
        {"columns": [], "feature_range": [1, 1]},
        {"columns": [], "feature_range": [0, float("inf")]},
        {"columns": [], "unexpected": True},
    ],
)
def test_minmax_empty_identity_requires_an_explicit_supported_recipe(config):
    """An empty artifact must not hide missing learned columns or invalid scale options."""
    assert get_inference_capability("MinMaxScaler", config, {}, engine="pandas") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_minmax_all_null_statistics_remain_unknown_after_pickle(engine):
    """Native NaN coefficients remain unusable learned state rather than a certified identity."""
    frame = pd.DataFrame({"x": [float("nan"), float("nan")]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["x"]}
    state = NodeRegistry.get_calculator("MinMaxScaler")().fit(frame, config)
    state = pickle.loads(pickle.dumps(state))
    assert state and all(
        np.isnan(state[name][0]) for name in ("min", "scale", "data_min", "data_max")
    )
    assert get_inference_capability("MinMaxScaler", config, state, engine=engine) is None
    applier = NodeRegistry.get_applier("MinMaxScaler")()
    record = {
        "name": "null",
        "type": "MinMaxScaler",
        "artifact": state,
        "params": config,
        "applier": applier,
    }
    recipe = {"name": "null", "transformer": "MinMaxScaler", "params": config}
    with pytest.raises(ValueError, match="finite numeric"):
        _inspect_step(record, recipe)
    output = applier.apply(frame, state)
    assert (output["x"].isna() if engine == "pandas" else output["x"].is_null()).all()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_minmax_identity_keeps_real_pipeline_worker_boundary(tmp_path, engine, monkeypatch):
    """Loaded disabled scalers remain exact local identities with the existing fitted-engine guard."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "empty", "transformer": "MinMaxScaler", "params": {"columns": []}}
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {
                    "n_estimators": 2,
                    "max_depth": 2,
                    "random_state": 42,
                    "n_jobs": 1,
                },
            },
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame[:0]), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "model")

    def forbidden(*args, **kwargs):
        """Loading an intentionally disabled scaler must never fit it again."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator("MinMaxScaler"), "fit", forbidden)
    restored = load_local_pipeline(tmp_path / "model")
    sample = (
        frame.drop(columns=["target"]) if isinstance(frame, pd.DataFrame) else frame.drop("target")
    )
    report = probe_fitted_preprocessing(restored, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed" and report["steps"][0]["context"] == "row"
    assert report["steps"][0]["state_validation"] == "node_owned"
    np.testing.assert_array_equal(restored.pipeline.predict(sample), pipeline.predict(sample))
    if engine == "pandas":
        assert require_partition_safe_pipeline(restored).steps[0].node_type == "MinMaxScaler"
        restored.pipeline.config["preprocessing"][0]["params"]["feature_range"] = [-1, 1]
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(restored)
    else:
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(restored)
