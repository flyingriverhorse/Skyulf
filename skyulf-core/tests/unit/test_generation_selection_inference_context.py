"""Generated features and selector facades inspect and replay their actual saved state."""

import pickle
from copy import deepcopy
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values=None):
    """Preserve native numeric schemas and nontrivial pandas row labels."""
    frame = pd.DataFrame(values or {"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.0] * 6})
    frame.index = [9, 2, 9, -1, 8, 0][: len(frame)]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, config, train=None):
    """Capture actual fit output, including supervised selection and facade dispatch."""
    frame = _frame(engine) if train is None else train
    target = pd.Series([0, 0, 0, 1, 1, 1], name="target")
    fit_data = frame if node.startswith("Feature") else (frame, target)
    return {
        "name": "features",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(fit_data, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,config",
    [
        ("FeatureGeneration", {"operations": [{"method": "add", "input_columns": ["a", "b"]}]}),
        ("ModelBasedSelection", {"columns": ["a", "b"]}),
        ("feature_selection", {"columns": ["a", "b"], "method": "variance"}),
    ],
)
def test_generation_and_selection_metadata_uses_pure_saved_state(node, config, engine, monkeypatch):
    """Metadata must inspect fitted artifacts without calculator fit or applier execution."""
    record = _record(node, engine, config)
    state = record["artifact"]
    before = artifact_digest(state)

    def forbidden(*args, **kwargs):
        """Expose accidental execution while querying metadata."""
        raise AssertionError("Metadata must not fit or apply")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier(node), "apply", forbidden)
    assert record["applier"].validate_inference_state(state) is state
    capability = get_inference_capability(node, config, state, engine=engine)
    assert capability is not None
    assert (capability.execution_kind, capability.context, capability.row_effect) == (
        "local",
        "row",
        "preserve",
    )
    assert get_inference_capability(node, config, state, engine="spark") is None
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", engine, config=config, execution_kind="python_batch")
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["VarianceThreshold", "UnivariateSelection"])
def test_existing_selection_children_accept_genuine_numpy_names(node, engine):
    """Facade validation must retain native names already accepted by real child fit/apply."""
    config = {"columns": (np.str_("a"), np.str_("b")), "k": 1}
    record = _record(node, engine, config)
    sample = _frame(engine)
    output = record["applier"].apply(sample, record["artifact"])
    assert list(output.columns) == ["a"]
    assert record["applier"].validate_inference_state(record["artifact"]) is record["artifact"]


def _assert_equal(actual, expected):
    """Keep schema, values and row labels exact when replaying a saved recipe."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


def _slice(frame, positions):
    """Select native row positions while retaining pandas labels."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _probe(record, frame, engine, monkeypatch):
    """Run the existing strict diagnostic after disabling every involved calculator."""
    before = artifact_digest(record)
    original = deepcopy(frame)

    def forbidden(*args, **kwargs):
        """No generated feature or selector may relearn at inference."""
        raise AssertionError("Unexpected inference fit")

    for node in (
        record["type"],
        "VarianceThreshold",
        "CorrelationThreshold",
        "UnivariateSelection",
        "ModelBasedSelection",
    ):
        monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    detail, output = _probe_step(
        record,
        {"name": record["name"], "transformer": record["type"], "params": record["params"]},
        frame,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    _assert_equal(frame, original)
    assert artifact_digest(record) == before
    assert detail["state_validation"] == "node_owned", detail
    return detail, output


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["FeatureGeneration", "FeatureMath", "FeatureGenerationNode"])
def test_feature_aliases_replay_ordered_arithmetic_ratio_and_collision_rules(
    node, engine, monkeypatch
):
    """Ordered operations must keep saved arithmetic and naming behavior across partition sizes."""
    operations: list[dict[str, Any]] = [
        {"method": method, "input_columns": ["a", "b"], "output_column": method}
        for method in ("add", "subtract", "multiply", "divide")
    ]
    operations += [
        {
            "operation_type": "ratio",
            "input_columns": ["add"],
            "secondary_columns": ["b"],
            "output_column": "ratio",
        },
        {
            "method": "add",
            "input_columns": ["add"],
            "constants": [np.float64(1)],
            "output_column": "add",
        },
    ]
    record = _record(node, engine, {"operations": operations, "epsilon": 0.5})
    sample = _frame(engine, {"a": [None, 2.0, 8.0, -1.0], "b": [2.0, 0.0, -2.0, 4.0]})
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output["add"].to_list() == [2.0, 2.0, 6.0, 3.0]
    assert output["add_1"].to_list() == [3.0, 3.0, 7.0, 4.0]
    assert output["divide"].to_list() == [0.0, 4.0, -4.0, -0.25]
    assert list(output.columns) == [
        "a",
        "b",
        "add",
        "subtract",
        "multiply",
        "divide",
        "ratio",
        "add_1",
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["mean", "sum", "count", "min", "max", "std", "median"])
def test_group_aggregation_reuses_saved_training_values_after_generated_inputs(
    method, engine, monkeypatch
):
    """Inference must look up learned groups without aggregating scoring rows or target values."""
    train = _frame(
        engine, {"group": ["x", "x", "y", "y", None, None], "a": [1.0, 3.0, 10.0, 20.0, 5.0, 7.0]}
    )
    operations = [
        {
            "method": "multiply",
            "input_columns": ["a"],
            "constants": [2.0],
            "output_column": "double",
        },
        {
            "operation_type": "group_agg",
            "input_columns": ["group"],
            "secondary_columns": ["double"],
            "method": method,
            "output_column": "learned",
        },
    ]
    record = _record("FeatureGeneration", engine, {"operations": operations}, train)
    mapping = record["artifact"]["operations"][1]["group_agg_mapping"]
    sample = _frame(engine, {"group": ["x", "y", None, "new"], "a": [900.0, -100.0, None, 5.0]})
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert detail["context"] == "row"
    assert output["learned"].to_list()[:3] == [
        mapping["values"][0],
        mapping["values"][1],
        mapping["null_value"],
    ]
    assert pd.isna(output["learned"].to_list()[3])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_aggregation_unresolved_fit_is_noop_but_unfitted_legacy_is_rejected(
    engine, monkeypatch
):
    """A saved unresolved lookup is distinct from the forbidden live-aggregation fallback."""
    record = _record(
        "FeatureGeneration",
        engine,
        {
            "operations": [
                {
                    "operation_type": "group_agg",
                    "input_columns": ["missing"],
                    "secondary_columns": ["a"],
                }
            ]
        },
    )
    assert record["artifact"]["operations"][0]["group_agg_mapping"] is None
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed", detail
    _assert_equal(output, _frame(engine))
    record["artifact"]["operations"][0].pop("group_agg_mapping")
    with pytest.raises(ValueError, match="refit"):
        record["applier"].validate_inference_state(record["artifact"])
    with pytest.raises(ValueError, match="refit"):
        record["applier"].apply(_frame(engine), record["artifact"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_datetime_generation_retains_native_null_dependent_dtype_diagnostics(engine, monkeypatch):
    """Calendar values remain native while strict diagnostics retain pandas dtype differences."""
    record = _record(
        "FeatureGeneration",
        engine,
        {
            "operations": [
                {
                    "operation_type": "datetime_extract",
                    "input_columns": ["date"],
                    "datetime_features": ["year", "is_weekend", "week"],
                    "output_prefix": "cal",
                }
            ]
        },
    )
    sample = _frame(engine, {"date": ["2024-01-01T00:30:00+02:00", "2025-02-01", "invalid", None]})
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["context"] == "row"
    assert [None if pd.isna(value) else value for value in output["cal_date_year"].to_list()] == [
        2023,
        2025,
        None,
        None,
    ]
    if engine == "pandas":
        assert detail["status"] == "failed"
        assert any(check.get("reason") == "output_mismatch" for check in detail["checks"])
    else:
        assert detail["status"] == "passed", detail


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_similarity_uses_fitted_backend_and_reports_pandas_rendering_context(engine, monkeypatch):
    """Datetime string rendering depends on neighboring pandas rows even with a pinned backend."""
    train = pd.DataFrame(
        {
            "a": pd.to_datetime(["2024-01-01", "2024-01-02 12:00"], format="mixed"),
            "b": ["2024-01-01", "2024-01-02"],
        }
    )
    frame = pl.from_pandas(train) if engine == "polars" else train
    record = _record(
        "FeatureGeneration",
        engine,
        {
            "operations": [
                {
                    "operation_type": "similarity",
                    "input_columns": ["a", "b"],
                    "output_column": "similarity",
                }
            ]
        },
        frame,
    )
    assert record["artifact"]["operations"][0]["similarity_backend"] in {"rapidfuzz", "difflib"}
    detail, _ = _probe(record, frame, engine, monkeypatch)
    full = record["applier"].apply(frame, record["artifact"])
    one = record["applier"].apply(_slice(frame, [0]), record["artifact"])
    if engine == "pandas":
        assert detail["status"] == "requires_context"
        assert full["similarity"].to_list()[0] != one["similarity"].to_list()[0]
    else:
        assert detail["status"] == "passed", detail
        _assert_equal(one, _slice(full, [0]))


def test_legacy_similarity_backend_remains_unknown_and_missing_dependency_rejected(monkeypatch):
    """Inspection must not silently substitute another similarity implementation."""
    from skyulf.preprocessing.feature_generation import _common

    record = _record(
        "FeatureGeneration",
        "pandas",
        {"operations": [{"operation_type": "similarity", "input_columns": ["a", "b"]}]},
    )
    op = record["artifact"]["operations"][0]
    op.pop("similarity_backend")
    assert (
        get_inference_capability("FeatureGeneration", {}, record["artifact"], engine="pandas")
        is None
    )
    op["similarity_backend"] = "rapidfuzz"
    monkeypatch.setattr(_common, "_HAS_RAPIDFUZZ", False)
    with pytest.raises(ImportError, match="rapidfuzz"):
        record["applier"].validate_inference_state(record["artifact"])


def test_nonnumeric_fill_is_usable_without_an_independent_partition_promise():
    """A pandas operation-level failure can omit an output only for batches containing nulls."""
    record = _record(
        "FeatureGeneration",
        "pandas",
        {
            "operations": [
                {"method": "add", "input_columns": ["a"], "fillna": "bad", "output_column": "out"}
            ]
        },
    )
    sample = _frame("pandas", {"a": [1.0, None]})
    assert (
        get_inference_capability("FeatureGeneration", {}, record["artifact"], engine="pandas")
        is None
    )
    full = record["applier"].apply(sample, record["artifact"])
    one = record["applier"].apply(sample.iloc[[0]], record["artifact"])
    assert "out" not in full.columns and "out" in one.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["select_from_model", "rfe"])
@pytest.mark.parametrize("drop", [True, np.bool_(False)])
def test_model_selection_replays_only_saved_membership(method, drop, engine, monkeypatch):
    """Scores and estimator choices must not rerun when inference values or ordering change."""
    config = {
        "columns": (np.str_("a"), np.str_("b")),
        "method": method,
        "drop_columns": drop,
        "n_features_to_select": 1,
    }
    record = _record("ModelBasedSelection", engine, config)
    record["artifact"]["feature_importances"] = {"a": np.nan, "b": np.inf}
    sample = _frame(
        engine, {"b": [None, 90.0, -8.0, 10.0], "keep": [1, 2, 3, 4], "a": [8.0, None, -4.0, 0.0]}
    )
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert list(output.columns) == (["keep", "a"] if drop else ["b", "keep", "a"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "method",
    ["variance", "correlation_threshold", "select_k_best", "select_from_model", "rfe", "unknown"],
)
def test_facade_replays_actual_child_states_without_relearning(method, engine, monkeypatch):
    """All four selector families and the genuine unknown-method no-op retain their own state."""
    config = {"columns": ["a", "b"], "method": method, "k": 1, "n_features_to_select": 1}
    record = _record("feature_selection", engine, config)
    sample = _frame(engine, {"b": [None, 90.0, -8.0, 10.0], "a": [8.0, None, -4.0, 0.0]})
    expected = record["applier"].apply(sample, record["artifact"])
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    _assert_equal(output, expected)


def test_facade_uses_child_context_and_preserves_unknown_saved_type_passthrough(monkeypatch):
    """The facade must not hard-code a row promise over its delegated owner's context."""
    record = _record("feature_selection", "pandas", {"method": "variance", "columns": ["a", "b"]})
    owner = NodeRegistry.get_applier("VarianceThreshold")
    monkeypatch.setattr(
        owner,
        "inference_capability",
        staticmethod(
            lambda state, *, engine: ExecutionCapability(
                engine, "apply", "local", "preserve", "global"
            )
        ),
    )
    capability = get_inference_capability(
        "feature_selection", {}, record["artifact"], engine="pandas"
    )
    assert capability is not None and capability.context == "global"
    legacy = {"type": "unknown_selector", "report": Decimal("1.5")}
    frame = _frame("pandas")
    assert record["applier"].validate_inference_state(legacy) is legacy
    assert record["applier"].apply(frame, legacy) is frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["FeatureGeneration", "ModelBasedSelection"])
def test_genuine_empty_operations_and_empty_model_selection_are_noops(node, engine, monkeypatch):
    """No-op states remain distinct from malformed learned selections."""
    config = {"operations": []} if node == "FeatureGeneration" else {"columns": []}
    record = _record(node, engine, config)
    sample = _frame(engine)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    _assert_equal(output, sample)


@pytest.mark.parametrize("damage", ["fields", "operation_type", "mapping_width", "mapping_values"])
def test_malformed_generated_state_is_rejected_without_aggregate_execution(damage):
    """An invalid saved lookup must fail inspection instead of reverting to live aggregation."""
    config = {
        "operations": [
            {"operation_type": "group_agg", "input_columns": ["b"], "secondary_columns": ["a"]}
        ]
    }
    record = _record("FeatureGeneration", "pandas", config)
    state = record["artifact"]
    if damage == "fields":
        state["extra"] = 1
    elif damage == "operation_type":
        state["operations"][0]["operation_type"] = "polynomial"
    elif damage == "mapping_width":
        state["operations"][0]["group_agg_mapping"]["values"] = []
    else:
        state["operations"][0]["group_agg_mapping"]["values"] = ["bad"]
    before = artifact_digest(state)
    with pytest.raises(ValueError):
        record["applier"].validate_inference_state(state)
    assert artifact_digest(state) == before


@pytest.mark.parametrize("node", ["ModelBasedSelection", "feature_selection"])
@pytest.mark.parametrize("damage", ["fields", "selected", "drop"])
def test_malformed_saved_membership_is_rejected_by_owner_and_facade(node, damage):
    """Delegated validation must reject corrupted membership before any columns are dropped."""
    record = _record(node, "pandas", {"method": "select_from_model", "columns": ["a", "b"]})
    state = record["artifact"]
    if damage == "fields":
        state["extra"] = 1
    elif damage == "selected":
        state["selected_columns"] = ["missing"]
    else:
        state["drop_columns"] = []
    before = artifact_digest(state)
    with pytest.raises(ValueError):
        record["applier"].validate_inference_state(state)
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("overwrite", [None, 1, Decimal("0")])
def test_generation_preserves_native_scalar_overwrite_flags(engine, overwrite):
    """Saved native truthiness must not be narrowed into a new boolean-only contract."""
    record = _record(
        "FeatureGeneration",
        engine,
        {
            "allow_overwrite": overwrite,
            "operations": [{"method": "add", "input_columns": ["a", "b"], "output_column": "a"}],
        },
    )
    state = record["artifact"]
    before = pickle.dumps(state)
    result = record["applier"].apply(_frame(engine), state)
    assert ("a_1" not in result.columns) == bool(overwrite)
    assert record["applier"].validate_inference_state(state) is state
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "method", ["variance", "correlation_threshold", "select_k_best", "select_from_model"]
)
def test_facade_children_preserve_native_scalar_drop_flags(method, engine):
    """Every delegated selector accepts actual no-drop states without coercing saved flags."""
    record = _record(
        "feature_selection",
        engine,
        {"columns": ["a", "b"], "method": method, "k": 1, "drop_columns": None},
    )
    state = record["artifact"]
    for flag in (None, Decimal("0")):
        # Each calculator stores this flag verbatim; both values keep all selected sources.
        state["drop_columns"] = flag
        before = pickle.dumps(state)
        result = record["applier"].apply(_frame(engine), state)
        _assert_equal(result, _frame(engine))
        assert record["applier"].validate_inference_state(state) is state
        assert pickle.dumps(state) == before
