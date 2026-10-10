"""Simple median context describes local replay without widening worker admission."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.core.portable_state import encode_state
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.partition_safety import _inspect_step, require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values):
    """Retain nullable numbers and duplicated row labels across native requests."""
    frame = pd.DataFrame({"x": pd.array(values, dtype="Int64"), "keep": [8, 3, 3, 1]})
    frame.index = [8, 3, 3, 1]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _equal(actual, expected):
    """Check native values, nulls, dtypes, columns and pandas index without tolerance."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_equal(actual, expected, check_exact=True)


def _take(frame, positions):
    """Use positions so duplicate pandas labels cannot change partition membership."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _fit(engine):
    """Obtain fractional learned medians from each engine's real calculator."""
    frame = _frame(engine, [1, 2, None, None])
    before = deepcopy(frame)
    config = {"columns": ["x"], "strategy": "median"}
    state = NodeRegistry.get_calculator("SimpleImputer")().fit(frame, config)
    _equal(frame, before)
    assert state["fill_values"] == {"x": 1.5}
    return config, state


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("values", [[None, 4, None, 7], [None] * 4])
def test_simple_median_local_context_replays_fitted_rows(engine, values):
    """Fractional medians preserve full, chunk, singleton, reordered and empty results."""
    config, state = _fit(engine)
    saved = pickle.dumps(state)
    frame = _frame(engine, values)
    before = deepcopy(frame)
    assert get_inference_capability("SimpleImputer", config, state, engine=engine) == (
        ExecutionCapability(
            engine, "apply", "local", "preserve", "row", config_match=(("strategy", "median"),)
        )
    )
    target = np.array([0, 1, 1, 0])
    applier = NodeRegistry.get_applier("SimpleImputer")()
    full, output_target = applier.apply((frame, target), state)
    assert output_target is target
    for positions in ([0], [1], [2], [3], [0, 1], [2, 3], [3, 2, 1, 0], []):
        _equal(applier.apply(_take(frame, positions), state), _take(full, positions))
    assert full["x"].to_list() == [1.5 if value is None else value for value in values]
    _equal(frame, before)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("fitted_engine", ["pandas", "polars"])
def test_simple_median_context_query_does_not_fit_or_apply(fitted_engine, monkeypatch):
    """Metadata reads inspect the saved artifact without data execution or mutation."""
    config, state = _fit(fitted_engine)
    saved = pickle.dumps((config, state))

    def forbidden(*args, **kwargs):
        """Fail if a metadata query attempts to learn or transform data."""
        raise AssertionError("Context query executed fit or apply")

    monkeypatch.setattr(NodeRegistry.get_calculator("SimpleImputer"), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier("SimpleImputer"), "apply", forbidden)
    for engine in ("pandas", "polars"):
        assert get_inference_capability("SimpleImputer", config, state, engine=engine) == (
            ExecutionCapability(
                engine, "apply", "local", "preserve", "row", config_match=(("strategy", "median"),)
            )
        )
    assert pickle.dumps((config, state)) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_simple_all_null_median_retains_native_artifact_boundary(engine):
    """Polars None statistics are inspectable while pandas empty fitted state stays unknown."""
    frame = _frame(engine, [None] * 4)
    config = {"columns": ["x"], "strategy": "median"}
    state = NodeRegistry.get_calculator("SimpleImputer")().fit(frame, config)
    saved = pickle.dumps(state)
    capability = get_inference_capability("SimpleImputer", config, state, engine=engine)
    if engine == "pandas":
        assert state == {} and capability is None
    else:
        assert state["fill_values"] == {"x": None}
        assert capability == ExecutionCapability(
            "polars", "apply", "local", "preserve", "row", config_match=(("strategy", "median"),)
        )
    applier = NodeRegistry.get_applier("SimpleImputer")()
    for positions in ([0, 1, 2, 3], [0], [1, 2], [3, 2, 1, 0], []):
        sample = _take(frame, positions)
        _equal(applier.apply(sample, state), sample)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("value", ["1.5", True, float("nan"), float("inf"), [1.5]])
def test_simple_median_rejects_non_numeric_or_non_finite_statistics(value):
    """A scalar mode value must not automatically become a valid numeric median."""
    config, state = _fit("polars")
    state["fill_values"]["x"] = value
    saved = pickle.dumps(state)
    for engine in ("pandas", "polars"):
        assert get_inference_capability("SimpleImputer", config, state, engine=engine) is None
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize(
    "change",
    [
        {"missing_counts": {"x": -1}, "total_missing": -1},
        {"missing_counts": {"x": True}, "total_missing": 1},
        {"total_missing": 3},
        {"fill_values": {}},
        {"columns": ["other"]},
        {"unexpected": 1},
    ],
)
def test_simple_median_keeps_existing_artifact_shape_and_count_validation(change):
    """Local median declarations must reject corruption accepted by neither saved-state path."""
    config, state = _fit("polars")
    state.update(change)
    assert get_inference_capability("SimpleImputer", config, state, engine="polars") is None


@pytest.mark.parametrize(
    "change",
    [{"strategy": "mean"}, {"columns": ["other"]}, {"fill_value": 7}, {"unexpected": 1}],
)
def test_simple_median_keeps_fitted_config_binding(change):
    """A reviewed artifact cannot be borrowed by a different inference recipe."""
    config, state = _fit("polars")
    config.update(change)
    assert get_inference_capability("SimpleImputer", config, state, engine="polars") is None


def test_simple_median_override_and_mismatched_mode_stay_unknown():
    """Recognizing the modal alias must not bypass callable identity or median recipe binding."""
    config, state = _fit("polars")
    applier = NodeRegistry.get_applier("SimpleImputer")()
    applier.apply = lambda *args: None
    assert (
        get_inference_capability("SimpleImputer", config, state, engine="polars", applier=applier)
        is None
    )
    config["strategy"] = "mode"
    assert get_inference_capability("SimpleImputer", config, state, engine="polars") is None


def test_simple_median_local_context_does_not_admit_workers_or_portable_codec():
    """Local row metadata grants neither Python worker execution nor portable median encoding."""
    config, state = _fit("polars")
    owner = SimpleImputerApplier
    normalized = owner.resolve_fitted_config(config, owner.validate_fitted_state(state))
    for engine, kind in (
        ("pandas", "python_batch"),
        ("polars", "python_batch"),
        ("polars", "native"),
        ("spark", "native"),
    ):
        with pytest.raises(UnsupportedExecutionError):
            require_capability(
                "SimpleImputer", "apply", engine, config=normalized, execution_kind=kind
            )
    assert get_inference_capability("SimpleImputer", config, state, engine="spark") is None
    with pytest.raises(ValueError, match="mean and constant only"):
        encode_state("SimpleImputer", state, max_bytes=4096)


@pytest.mark.parametrize("strategy", ["median", "most_frequent"])
def test_simple_local_statistics_keep_normalized_python_strategy(strategy):
    """Extending median inspection must preserve canonical scalar types of existing modal state."""
    _, state = _fit("polars")
    state["strategy"] = np.str_(strategy)
    saved = pickle.dumps(state)
    normalized = SimpleImputerApplier.validate_fitted_state(state)
    assert normalized["strategy"] == strategy
    assert type(normalized["strategy"]) is str
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["mean", "median", "mode", "constant"])
def test_simple_explicit_empty_selection_is_a_valid_saved_identity(engine, strategy):
    """Explicitly disabling imputation must preserve rows while keeping median workers excluded."""
    frame = _frame(engine, [1, None, 3, None])
    config = {"columns": [], "strategy": strategy}
    state = NodeRegistry.get_calculator("SimpleImputer")().fit(frame, config)
    assert state == {}
    state = pickle.loads(pickle.dumps(state))
    original = deepcopy(frame)
    before = pickle.dumps((config, state))
    capability = get_inference_capability("SimpleImputer", config, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert capability.execution_kind == (
        "python_batch" if engine == "pandas" and strategy != "median" else "local"
    )
    applier = SimpleImputerApplier()
    for positions in ([0, 1, 2, 3], [1], [3, 0, 1], []):
        sample = _take(frame, positions)
        _equal(applier.apply(sample, state), sample)
    record = {
        "name": "empty",
        "type": "SimpleImputer",
        "artifact": state,
        "params": config,
        "applier": applier,
    }
    recipe = {"name": "empty", "transformer": "SimpleImputer", "params": config}
    if strategy == "median":
        with pytest.raises(UnsupportedExecutionError):
            _inspect_step(record, recipe)
    else:
        assert _inspect_step(record, recipe).action == "apply"
    _equal(frame, original)
    assert pickle.dumps((config, state)) == before


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"columns": ["x"]},
        {"columns": [], "_auto_columns": True},
        {"columns": [], "strategy": "unknown"},
        {"columns": [], "strategy": "mode", "fill_value": 4},
        {"columns": [], "strategy": "median", "fill_value": 4},
        {"columns": [], "strategy": "mode", "unexpected": True},
    ],
)
def test_simple_empty_identity_requires_an_explicit_supported_recipe(config):
    """A missing artifact cannot conceal learned selection or unsupported recipe options."""
    assert get_inference_capability("SimpleImputer", config, {}, engine="pandas") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_simple_empty_mode_keeps_real_pipeline_worker_boundary(tmp_path, engine, monkeypatch):
    """A saved explicit identity can be inspected without admitting a Polars-fitted worker model."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "empty",
                    "transformer": "SimpleImputer",
                    "params": {"columns": [], "strategy": "mode"},
                }
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
        """Loading an intentionally disabled imputer must never learn new fill values."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator("SimpleImputer"), "fit", forbidden)
    restored = load_local_pipeline(tmp_path / "model")
    sample = frame.drop(columns=["target"]) if engine == "pandas" else frame.drop("target")
    report = probe_fitted_preprocessing(restored, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed" and report["steps"][0]["context"] == "row"
    assert report["steps"][0]["state_validation"] == "node_owned"
    np.testing.assert_array_equal(restored.pipeline.predict(sample), pipeline.predict(sample))
    if engine == "pandas":
        assert require_partition_safe_pipeline(restored).steps[0].node_type == "SimpleImputer"
        restored.pipeline.config["preprocessing"][0]["params"]["columns"] = ["x"]
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(restored)
    else:
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(restored)
