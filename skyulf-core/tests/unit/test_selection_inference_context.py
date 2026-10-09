"""Saved selector decisions are local, immutable, and independent of inference batches."""

from copy import deepcopy
from typing import Any

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.inference._fitted_contract import resolve_fitted_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

_NODES = ("CorrelationThreshold", "UnivariateSelection", "VarianceThreshold")


def _frame(engine):
    """Keep numeric candidates and untouched nullable text in a stable column order."""
    frame = pd.DataFrame(
        {
            "keep": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "redundant": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "constant": [7.0] * 6,
            "tag": ["a", None, "b", "a", "b", "c"],
        },
        index=[8, 3, 3, 1, 9, 2],
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


def _fit(node, engine="pandas", **overrides):
    """Fit actual selectors once, including an explicit target for univariate scoring."""
    config = {"columns": ["keep", "redundant", "constant"], **overrides}
    data = _frame(engine)
    if node == "UnivariateSelection":
        config = {"method": "select_k_best", "k": 2, "score_func": "f_regression", **config}
        config["problem_type"] = "regression"
        data = (data, pd.Series([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))
    state = NodeRegistry.get_calculator(node)().fit(data, config)
    return config, state


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", _NODES)
def test_selector_context_describes_saved_apply_without_worker_admission(node, engine):
    """Saved decisions must advertise row-local apply while strict workers still reject them."""
    config, state = _fit(node, engine)
    applier = NodeRegistry.get_applier(node)()
    capability = get_inference_capability(node, config, state, engine=engine, applier=applier)
    assert capability is not None
    assert (capability.context, capability.row_effect, capability.execution_kind) == (
        "row",
        "preserve",
        "local",
    )
    assert applier.validate_inference_state(state) is state
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", engine, config=config, execution_kind="python_batch")
    record = {
        "name": "select",
        "type": node,
        "params": config,
        "artifact": state,
        "applier": applier,
    }
    recipe = {"name": "select", "transformer": node, "params": config}
    with pytest.raises(ValueError, match="validation hooks are missing"):
        resolve_fitted_step(record, recipe)


def _assert_frames(actual, expected):
    """Compare values, nulls, schema, column order and the pandas row index."""
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    else:
        assert_polars_equal(actual, expected)


def _slice(frame, start, count):
    """Slice by position without losing duplicate pandas index labels."""
    return (
        frame.iloc[start : start + count]
        if isinstance(frame, pd.DataFrame)
        else frame.slice(start, count)
    )


def _forbidden(*args, **kwargs):
    """Fail if inference inspection attempts to learn or execute a fit-only callback."""
    raise AssertionError("Inference called fit or a reporting callback")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("drop_columns", [True, False])
@pytest.mark.parametrize("node", _NODES)
def test_saved_selector_apply_is_invariant_to_batch_and_row_order(
    node, engine, drop_columns, monkeypatch
):
    """Selectors must use saved decisions when values, nulls and batch neighbors change."""
    config, state = _fit(node, engine, drop_columns=drop_columns)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    sample = _frame("pandas")
    sample["constant"] = [0.0, 7.0, None, 300.0, -1.0, 2.0]
    sample.iloc[1, 0] = float("nan")
    sample["redundant"] = [5.0, 1.0, -3.0, 9.0, 2.0, None]
    sample = pl.from_pandas(sample) if engine == "polars" else sample
    original = deepcopy(sample)
    before = artifact_digest(state)
    columns = list(sample.columns)
    if drop_columns:
        columns.remove("redundant" if node == "CorrelationThreshold" else "constant")
    expected = sample[columns]
    applier = NodeRegistry.get_applier(node)()
    assert get_inference_capability(node, config, state, engine=engine) is not None
    for _ in range(2):
        _assert_frames(applier.apply(sample, state), expected)
    for size in (1, 2, 4):
        parts = [
            applier.apply(_slice(sample, start, size), state)
            for start in range(0, len(sample), size)
        ]
        actual = pl.concat(parts) if engine == "polars" else pd.concat(parts)
        _assert_frames(actual, expected)
    _assert_frames(applier.apply(sample[::-1], state), expected[::-1])
    _assert_frames(applier.apply(_slice(sample, 0, 0), state), _slice(expected, 0, 0))
    _assert_frames(sample, original)
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", _NODES)
def test_empty_fitted_selector_artifacts_are_valid_noops(node, engine, monkeypatch):
    """An actual fit with no candidate columns must retain its empty passthrough state."""
    frame = pd.DataFrame({"tag": ["a", None, "b"]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = NodeRegistry.get_calculator(node)().fit(frame, {})
    assert state == {}
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    applier = NodeRegistry.get_applier(node)()
    assert applier.validate_inference_state(state) is state
    assert get_inference_capability(node, {}, state, engine=engine) is not None
    _assert_frames(applier.apply(frame, state), frame)
    assert state == {}


@pytest.mark.parametrize("node", _NODES)
@pytest.mark.parametrize(
    "mutation",
    [
        "not_dict",
        "wrong_type",
        "extra",
        "missing",
        "list_type",
        "duplicate",
        "nonstring",
        "drop_flag",
    ],
)
def test_selector_context_rejects_malformed_saved_decisions(node, mutation):
    """Malformed decisions must fail inspection before they can masquerade as valid noops."""
    config, state = _fit(node)
    field = "columns_to_drop" if node == "CorrelationThreshold" else "selected_columns"
    changes = {
        "not_dict": None,
        "wrong_type": {**state, "type": "wrong"},
        "extra": {**state, "unknown_behavior": True},
        "missing": {key: value for key, value in state.items() if key != field},
        "list_type": {**state, field: tuple(state[field])},
        "duplicate": {**state, field: ["keep", "keep"]},
        "nonstring": {**state, field: [None]},
        "drop_flag": {**state, "drop_columns": "false"},
    }
    malformed: Any = changes[mutation]
    applier: Any = NodeRegistry.get_applier(node)
    with pytest.raises(ValueError):
        applier.validate_inference_state(malformed)
    with pytest.raises(ValueError):
        get_inference_capability(node, config, malformed, engine="pandas")


@pytest.mark.parametrize("node", ["UnivariateSelection", "VarianceThreshold"])
@pytest.mark.parametrize(
    "changes",
    [
        {"selected_columns": ["absent"]},
        {"selected_columns": None},
        {"candidate_columns": ["keep", "keep"]},
        {"candidate_columns": None},
    ],
)
def test_selected_columns_must_belong_to_unique_candidates(node, changes):
    """Selection membership must be coherent before the shared apply helper uses sets."""
    _, state = _fit(node)
    applier: Any = NodeRegistry.get_applier(node)
    with pytest.raises(ValueError):
        applier.validate_inference_state({**state, **changes})


@pytest.mark.parametrize("node", _NODES)
def test_selector_capabilities_abstain_for_other_engines(node):
    """Local row context must never claim Spark or unknown-engine execution."""
    applier: Any = NodeRegistry.get_applier(node)
    assert applier.inference_capability(None, engine="spark") is None
    assert applier.inference_capability(None, engine="other") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_univariate_missing_target_artifact_keeps_legacy_report_fields(engine):
    """The genuine no-target schema must be accepted without changing its stored keys."""
    frame = _frame(engine)
    config = {"allow_missing_target": True, "columns": ["keep", "redundant"], "k": 1}
    state = NodeRegistry.get_calculator("UnivariateSelection")().fit(frame, config)
    assert "scores" in state and "pvalues" in state
    applier = NodeRegistry.get_applier("UnivariateSelection")()
    before = artifact_digest(state)
    assert applier.validate_inference_state(state) is state
    _assert_frames(applier.apply(frame, state), frame)
    assert artifact_digest(state) == before


@pytest.mark.parametrize("node", _NODES)
def test_fit_only_reporting_metadata_does_not_change_apply_decisions(node):
    """Nonfinite reports and callable correlation metadata are harmless during saved apply."""
    config, state = _fit(node)
    if node == "CorrelationThreshold":
        state.update({"threshold": float("nan"), "method": _forbidden})
    elif node == "VarianceThreshold":
        state.update({"threshold": float("inf"), "variances": {"constant": float("nan")}})
    else:
        state.update({"feature_scores": {"keep": float("inf")}, "p_values": {"keep": float("nan")}})
    applier = NodeRegistry.get_applier(node)()
    before = artifact_digest(state)
    assert applier.validate_inference_state(state) is state
    assert get_inference_capability(node, config, state, engine="pandas") is not None
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_variance_empty_selection_keeps_uncandidate_columns(engine):
    """Dropping every constant candidate must preserve row identity and unrelated columns."""
    frame = pd.DataFrame({"constant": [7.0, 7.0, 7.0], "tag": [None, "a", "b"]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["constant"]}
    state = NodeRegistry.get_calculator("VarianceThreshold")().fit(frame, config)
    assert state["selected_columns"] == []
    applier = NodeRegistry.get_applier("VarianceThreshold")()
    assert applier.validate_inference_state(state) is state
    _assert_frames(applier.apply(frame, state), frame[["tag"]])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["UnivariateSelection", "VarianceThreshold"])
def test_empty_selected_schema_preserves_rows(node, engine):
    """Removing every candidate must retain row count even with zero output columns."""
    frame = pd.DataFrame({"constant": [7.0, 7.0, 7.0]}, index=[3, 3, 1])
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["constant"]}
    data = frame
    if node == "UnivariateSelection":
        config.update({"k": 0, "score_func": "r_regression", "problem_type": "regression"})
        data = (frame, pd.Series([0.0, 1.0, 2.0]))
    state = NodeRegistry.get_calculator(node)().fit(data, config)
    assert state["selected_columns"] == []
    if node == "UnivariateSelection":
        assert state["p_values"] == {}
    applier = NodeRegistry.get_applier(node)()
    assert applier.validate_inference_state(state) is state
    result = applier.apply(frame, state)
    assert result.shape == (3, 0)
    if engine == "pandas":
        assert list(result.index) == [3, 3, 1]
    capability = get_inference_capability(node, config, state, engine=engine)
    assert capability is not None and capability.row_effect == "preserve"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_real_all_missing_variance_report_remains_valid(engine):
    """A fitted undefined variance is reporting data and must not invalidate saved decisions."""
    frame = pd.DataFrame({"missing": [float("nan")] * 3, "varying": [1.0, 2.0, 3.0]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["missing", "varying"]}
    with pytest.warns(RuntimeWarning):
        state = NodeRegistry.get_calculator("VarianceThreshold")().fit(frame, config)
    assert pd.isna(state["variances"]["missing"])
    applier = NodeRegistry.get_applier("VarianceThreshold")()
    assert applier.validate_inference_state(state) is state
    _assert_frames(applier.apply(frame, state), frame[["varying"]])
