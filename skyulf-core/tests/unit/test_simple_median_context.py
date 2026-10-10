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
