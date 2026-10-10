"""Missing-category encoding reuses saved local rules without granting worker admission."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal
from sklearn.preprocessing import OneHotEncoder

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

MISSING = "__mlops_missing__"
ESCAPE = "__mlops_literal__:"
LITERALS = [MISSING, ESCAPE + MISSING, ESCAPE + ESCAPE + MISSING, "red"]
VALUES = [None, *LITERALS, None]


def _frame(engine, kind="plain", values=VALUES):
    """Retain native textual dtypes and duplicate pandas labels around missing values."""
    if engine == "pandas":
        dtype = {"plain": object, "category": "category", "string": "string"}[kind]
        frame = pd.DataFrame({"color": pd.Series(values, dtype=dtype)})
        frame.index = [7, 2, 2, 0, 1, 3][: len(values)]
        return frame
    dtype = {"plain": pl.String, "category": pl.Categorical, "enum": pl.Enum(LITERALS)}[kind]
    return pl.DataFrame({"color": pl.Series(values, dtype=dtype)})


def _fitted(engine, kind="plain", options=None):
    """Fit the actual registered owner with its existing escaping policy."""
    frame = _frame(engine, kind)
    config = {"columns": ["color"], "include_missing": True, **(options or {})}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    return frame, config, state


def _take(frame, positions):
    """Partition positionally without losing duplicate row labels or native schemas."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _equal(actual, expected):
    """A row contract includes exact values, dtypes, column order and row labels."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize(
    "engine,kind",
    [
        ("pandas", "plain"),
        ("pandas", "category"),
        ("pandas", "string"),
        ("polars", "plain"),
        ("polars", "category"),
        ("polars", "enum"),
    ],
)
@pytest.mark.parametrize(
    "options", [{"max_categories": None}, {}, {"max_categories": 3, "drop_first": True}]
)
def test_missing_encoding_reports_local_context_and_exact_native_replay(engine, kind, options):
    """Missing and reserved text must retain their saved identities in every request partition."""
    frame, config, state = _fitted(engine, kind, options)
    original = deepcopy(frame)
    saved = pickle.dumps(state)
    assert get_inference_capability(
        "OneHotEncoder", config, state, engine=engine
    ) == ExecutionCapability(engine, "apply", "local", "preserve", "row")
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    target = np.arange(len(frame))
    full, output_target = applier.apply((frame, target), state)
    assert output_target is target
    if options.get("max_categories") is None:
        assert np.unique(full.to_numpy(), axis=0).shape[0] == 5
    for positions in ([0], [1], [2], [5, 3, 0, 1], [], list(range(5, -1, -1))):
        _equal(applier.apply(_take(frame, positions), state), _take(full, positions))
    _equal(frame, original)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_unversioned_missing_encoding_retains_legacy_row_context(engine):
    """Inspecting an old artifact must not upgrade its deliberate missing/literal collision."""
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int8, handle_unknown="error").fit(
        np.array([[MISSING], ["red"]], dtype=object)
    )
    config = {
        "columns": ["color"],
        "include_missing": True,
        "max_categories": None,
        "handle_unknown": "error",
    }
    state = {
        "type": "onehot",
        "columns": ["color"],
        "encoder_object": encoder,
        "feature_names": encoder.get_feature_names_out(["color"]).tolist(),
        "prefix_separator": "_",
        "drop_original": True,
        "include_missing": True,
    }
    saved = pickle.dumps(state)
    assert get_inference_capability(
        "OneHotEncoder", config, state, engine=engine
    ) == ExecutionCapability(engine, "apply", "local", "preserve", "row")
    output = NodeRegistry.get_applier("OneHotEncoder")().apply(
        _frame(engine, values=[None, MISSING, "red"]), state
    )
    np.testing.assert_array_equal(output.to_numpy(), [[1, 0], [1, 0], [0, 1]])
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("version", [None, True, False, 0, 2, "1", 1.0, np.int64(1)])
def test_missing_context_keeps_raw_version_type_validation(version):
    """Scalar normalization must not turn an invalid apply policy into an accepted declaration."""
    frame, config, state = _fitted("pandas")
    state["missing_encoding_version"] = version
    assert get_inference_capability("OneHotEncoder", config, state, engine="pandas") is None
    with pytest.raises(ValueError, match="missing encoding version"):
        NodeRegistry.get_applier("OneHotEncoder")().apply(frame, state)


@pytest.mark.parametrize(
    "engine,native_null", [("pandas", False), ("polars", False), ("polars", True)]
)
def test_all_missing_training_retains_local_row_context(engine, native_null):
    """An all-null fit still learns one reusable missing category and preserves empty schemas."""
    frame = (
        pl.DataFrame({"color": pl.Series([None] * 6, dtype=pl.Null)})
        if native_null
        else _frame(engine, values=[None] * 6)
    )
    config = {"columns": ["color"], "include_missing": True}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    full = applier.apply(frame, state)
    np.testing.assert_array_equal(full.to_numpy(), np.ones((6, 1)))
    for positions in ([0], [5, 2, 0], []):
        _equal(applier.apply(_take(frame, positions), state), _take(full, positions))


@pytest.mark.parametrize("change", ["recipe", "disabled", "integer_flag", "extra"])
def test_missing_context_retains_config_and_artifact_checks(change):
    """Escaping metadata cannot authorize mismatched recipes or unsupported saved fields."""
    _, config, state = _fitted("pandas")
    if change == "recipe":
        config["include_missing"] = False
    elif change == "disabled":
        state["include_missing"] = False
        config["include_missing"] = False
    elif change == "integer_flag":
        state["include_missing"] = 1
    else:
        state["future_missing_policy"] = "changed"
    assert get_inference_capability("OneHotEncoder", config, state, engine="pandas") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["ignore", "error"])
def test_unseen_missing_value_keeps_native_policy(engine, policy):
    """A row declaration cannot turn an unseen missing category into a known training category."""
    frame = _frame(engine, values=["red", MISSING])
    config = {"columns": ["color"], "include_missing": True, "handle_unknown": policy}
    state = NodeRegistry.get_calculator("OneHotEncoder")().fit(frame, config)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    sample = _frame(engine, values=[None])
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    if policy == "error":
        with pytest.raises(ValueError, match="unknown categor"):
            applier.apply(sample, state)
    else:
        assert applier.apply(sample, state).to_numpy().sum() == 0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("cap", [None, 20])
def test_missing_context_is_pure_and_stays_outside_workers(engine, cap, monkeypatch):
    """Neither missing policy may execute data or gain admission through uncapped declarations."""
    _, config, state = _fitted(engine, options={"max_categories": cap})
    saved = pickle.dumps(state)

    def forbidden(*args, **kwargs):
        """Fail whenever metadata inspection attempts to execute learned transformations."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(NodeRegistry.get_calculator("OneHotEncoder"), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier("OneHotEncoder"), "apply", forbidden)
    monkeypatch.setattr(type(state["encoder_object"]), "transform", forbidden)
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    for worker, kind in (
        ("pandas", "python_batch"),
        ("polars", "python_batch"),
        ("spark", "native"),
    ):
        with pytest.raises(UnsupportedExecutionError):
            require_capability("OneHotEncoder", "apply", worker, config=config, execution_kind=kind)
    assert pickle.dumps(state) == saved
