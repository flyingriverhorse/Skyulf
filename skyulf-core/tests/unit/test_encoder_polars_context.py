"""Reviewed encoder and interaction context remains separate from worker admission."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _fitted(node, engine, options=None):
    """Fit real native state with duplicate labels, nulls and unmodified companion columns."""
    frame = pd.DataFrame(
        {"category": ["b", "a", None, "b"], "a": [1.5, -2.0, 0.0, None], "b": [3, 2, -4, 1]},
        index=[8, 3, 3, 1],
    )
    config = (
        {"columns": ["category"], "max_categories": None}
        if node == "OneHotEncoder"
        else {"columns": ["b", "a"]}
    )
    config.update(options or {})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    return frame, config, state


def _equal(actual, expected):
    """Compare values, dtypes, column order and row labels without numeric tolerance."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_equal(actual, expected, check_exact=True)


def _take(frame, positions):
    """Slice by position while retaining duplicate pandas index labels."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,options",
    [
        ("OneHotEncoder", {}),
        ("OneHotEncoder", {"drop_first": True}),
        ("OneHotEncoder", {"drop_original": False}),
        ("OneHotEncoder", {"handle_unknown": "error"}),
        ("FeatureInteraction", {}),
        ("FeatureInteraction", {"degree": 3, "interaction_only": False}),
        ("FeatureInteraction", {"degree": 4, "interaction_only": False, "include_bias": True}),
        ("FeatureInteraction", {"degree": 4, "include_bias": True}),
    ],
)
def test_reviewed_encoding_context_replays_saved_rows(engine, node, options):
    """The reviewed declaration must match exact full, singleton, reordered and empty execution."""
    frame, config, state = _fitted(node, engine, options)
    original = deepcopy(frame)
    state_bytes = pickle.dumps(state)
    expected_capability = ExecutionCapability(
        engine,
        "apply",
        "local" if engine == "polars" else "python_batch",
        "preserve",
        "row",
        config_match=(("max_categories", None), ("include_missing", False))
        if node == "OneHotEncoder" and engine == "pandas"
        else (),
    )
    assert get_inference_capability(node, config, state, engine=engine) == expected_capability
    target = np.array([0, 1, 1, 0])
    applier = NodeRegistry.get_applier(node)()
    full, output_target = applier.apply((frame, target), state)
    assert output_target is target
    for positions in ([0], [1], [2], [3], [3, 2, 1, 0], []):
        _equal(applier.apply(_take(frame, positions), state), _take(full, positions))
    _equal(frame, original)
    assert pickle.dumps(state) == state_bytes


@pytest.mark.parametrize("node", ["OneHotEncoder", "FeatureInteraction"])
def test_encoding_context_query_executes_no_fit_or_apply(node, monkeypatch):
    """Metadata queries inspect saved state without triggering native execution or new learning."""
    _, config, state = _fitted(node, "polars")
    state_bytes = pickle.dumps(state)

    def forbidden(*args, **kwargs):
        """Make accidental data execution fail instead of hiding it behind a context assertion."""
        raise AssertionError("Metadata must not execute transformations")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier(node), "apply", forbidden)
    assert get_inference_capability(node, config, state, engine="polars") == ExecutionCapability(
        "polars", "apply", "local", "preserve", "row"
    )
    assert pickle.dumps(state) == state_bytes


@pytest.mark.parametrize("node", ["OneHotEncoder", "FeatureInteraction"])
@pytest.mark.parametrize("change", ["names", "extra", "config", "option", "override"])
def test_encoding_context_keeps_saved_validation_and_applier_identity(node, change):
    """A local declaration must not admit corrupted state, changed configuration or callbacks."""
    _, config, state = _fitted(node, "polars")
    applier = NodeRegistry.get_applier(node)()
    if change == "names":
        state["feature_names"] = ["changed"]
    elif change == "extra":
        state["ignored"] = True
    elif change == "config":
        config["columns"] = ["absent"]
    elif change == "option":
        config["drop_first" if node == "OneHotEncoder" else "include_bias"] = True
    else:
        applier.apply = lambda *args: None
    assert get_inference_capability(node, config, state, engine="polars", applier=applier) is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_onehot_explicit_empty_selection_declares_row_context(engine):
    """An explicit no-op keeps its input rows without needing a fitted encoder."""
    _, config, state = _fitted("OneHotEncoder", engine, {"columns": []})
    capability = get_inference_capability("OneHotEncoder", config, state, engine=engine)
    assert capability is not None and capability.context == "row"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["ignore", "error"])
def test_onehot_declared_context_retains_unknown_category_policy(engine, policy):
    """Row context must not hide the encoder's saved error policy for unseen categories."""
    frame, config, state = _fitted("OneHotEncoder", engine, {"handle_unknown": policy})
    sample = frame.iloc[[0]].copy() if isinstance(frame, pd.DataFrame) else frame.head(1)
    if isinstance(sample, pd.DataFrame):
        sample["category"] = "unseen"
    else:
        sample = sample.with_columns(pl.lit("unseen").alias("category"))
    assert get_inference_capability("OneHotEncoder", config, state, engine=engine) is not None
    applier = NodeRegistry.get_applier("OneHotEncoder")()
    if policy == "error":
        with pytest.raises(ValueError, match="unknown categories"):
            applier.apply(sample, state)
    else:
        result = applier.apply(sample, state)
        assert sum(result[name].to_list()[0] for name in state["feature_names"]) == 0


@pytest.mark.parametrize("node", ["OneHotEncoder", "FeatureInteraction"])
def test_encoding_local_context_does_not_admit_other_workers(node):
    """Polars local context cannot broaden Python batch or native Spark execution admission."""
    _, config, state = _fitted(node, "polars")
    for engine, kind in (("polars", "python_batch"), ("polars", "native"), ("spark", "native")):
        with pytest.raises(UnsupportedExecutionError):
            require_capability(node, "apply", engine, config=config, execution_kind=kind)
    assert get_inference_capability(node, config, state, engine="spark") is None
