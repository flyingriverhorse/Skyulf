"""Focused unit tests for
``backend.ml_pipeline._execution._leakage_validation``.

These build small, synthetic ``NodeConfig`` graphs directly (no dataset,
no execution) to exercise the pre-execution leakage guard in isolation —
this is the fastest, most reliable way to *trigger* the
"Data leakage risk" ``ValueError`` deliberately, without needing a real
dataset or a full pipeline run.
"""

import logging

import pytest

from backend.ml_pipeline._execution._leakage_validation import (
    NO_SPLIT_DIAGNOSTIC,
    validate_no_preprocessing_before_split,
)
from backend.ml_pipeline._execution.schemas import NodeConfig


def _node(
    node_id: str, step_type: str, inputs: list[str], params: dict | None = None
) -> NodeConfig:
    """Builds a minimal ``NodeConfig`` for graph-shape-only tests."""
    return NodeConfig(node_id=node_id, step_type=step_type, params=params or {}, inputs=inputs)


def test_raises_when_scaler_precedes_splitter():
    """A StandardScaler wired before the TrainTestSplitter must be blocked."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("scale", "StandardScaler", ["load"]),
        _node("split", "TrainTestSplitter", ["scale"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_allows_scaler_after_splitter():
    """The same node, moved after the split, is the correct/safe order."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("split", "TrainTestSplitter", ["load"]),
        _node("scale", "StandardScaler", ["split"]),
        _node("model", "LogisticRegression", ["scale"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_no_splitter_logs_explicit_diagnostic(caplog):
    """Pipelines with no train/test boundary get an explicit diagnostic
    instead of silence (G1) — still non-blocking."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("impute", "SimpleImputer", ["load"]),
        _node("scale", "StandardScaler", ["impute"]),
    ]
    with caplog.at_level(
        logging.WARNING, logger="backend.ml_pipeline._execution._leakage_validation"
    ):
        validate_no_preprocessing_before_split(nodes)  # must not raise
    assert any("No train/test split" in r.message for r in caplog.records)


def test_no_splitter_silent_under_ignore(caplog):
    """on_leakage='ignore' suppresses even the advisory diagnostic."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("impute", "SimpleImputer", ["load"]),
    ]
    with caplog.at_level(
        logging.WARNING, logger="backend.ml_pipeline._execution._leakage_validation"
    ):
        validate_no_preprocessing_before_split(nodes, on_leakage="ignore")
    assert caplog.records == []


def test_feature_target_split_does_not_trigger_leakage():
    """FeatureTargetSplitter only separates X/y - no train/test boundary."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("encode", "OneHotEncoder", ["load"]),
        _node("split_xy", "feature_target_split", ["encode"]),
        _node("model", "LogisticRegression", ["split_xy"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_stateless_nodes_before_splitter_are_allowed():
    """Rule-based/stateless nodes (fixed bounds) never leak."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("bounds", "ManualBounds", ["load"]),
        _node("split", "TrainTestSplitter", ["bounds"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


@pytest.mark.parametrize(
    ("step_type", "params"),
    [
        ("HashEncoder", {}),
        ("MissingIndicator", {}),
        # Param-less DropMissingColumns is the explicit/no-op mode (exempt);
        # its data-dependent mode is the positive missing-% threshold.
        ("DropMissingColumns", {"missing_threshold": 50}),
        ("Deduplicate", {}),
    ],
)
def test_reclassified_stateful_nodes_before_splitter_are_blocked(step_type, params):
    """F-16: nodes previously exempted as 'stateless' do learn from the data
    they are fitted on (hash bucket occupancy, missingness structure, drop
    lists, duplicate sets) and are now gated."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("step", step_type, ["load"], params=params),
        _node("split", "TrainTestSplitter", ["step"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_step_type_lists_are_derived_from_the_core_registry():
    """G2: the backend gate consumes the skyulf-core registry-derived lists;
    there is no second hand-maintained copy to drift."""
    from backend.ml_pipeline._execution import _leakage_validation
    from skyulf.leakage import data_dependent_transformers, train_test_splitters

    assert _leakage_validation.data_dependent_step_types() == data_dependent_transformers()
    assert _leakage_validation.train_test_split_step_types() == train_test_splitters()


def test_on_leakage_warn_logs_instead_of_raising(caplog):
    """on_leakage='warn' restores the old non-blocking behaviour."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("scale", "StandardScaler", ["load"]),
        _node("split", "TrainTestSplitter", ["scale"]),
    ]
    with caplog.at_level(
        logging.WARNING, logger="backend.ml_pipeline._execution._leakage_validation"
    ):
        validate_no_preprocessing_before_split(nodes, on_leakage="warn")  # must not raise
    assert any("Data leakage risk" in r.message for r in caplog.records)


def test_validate_returns_passed_verdict_for_a_clean_graph():
    """The gate returns a structured verdict the engine persists on the job."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("split", "TrainTestSplitter", ["load"]),
        _node("scale", "StandardScaler", ["split"]),
    ]
    verdict = validate_no_preprocessing_before_split(nodes)
    assert verdict == {"status": "passed", "messages": []}


def test_validate_returns_no_split_verdict_with_the_advisory_diagnostic(caplog):
    """A splitter-less graph gets the advisory diagnostic in the verdict."""
    nodes = [_node("load", "DataLoader", []), _node("scale", "StandardScaler", ["load"])]
    with caplog.at_level(
        logging.WARNING, logger="backend.ml_pipeline._execution._leakage_validation"
    ):
        verdict = validate_no_preprocessing_before_split(nodes)
    assert verdict["status"] == "no_split"
    assert verdict["messages"] == [NO_SPLIT_DIAGNOSTIC]


def test_validate_returns_warning_verdict_in_warn_mode():
    """warn mode returns the violation messages instead of raising."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("scale", "StandardScaler", ["load"]),
        _node("split", "TrainTestSplitter", ["scale"]),
    ]
    verdict = validate_no_preprocessing_before_split(nodes, on_leakage="warn")
    assert verdict["status"] == "warnings"
    assert len(verdict["messages"]) == 1
    assert "Data leakage risk" in verdict["messages"][0]


def test_raises_for_indirect_ancestor_through_branching_graph():
    """The leaking node need not be directly wired to the splitter -
    any path that reaches a splitter downstream counts."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("impute", "SimpleImputer", ["load"]),
        _node("clean", "ValueReplacement", ["impute"]),
        _node("split", "TrainTestSplitter", ["clean"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="'impute'"):
        validate_no_preprocessing_before_split(nodes)


@pytest.mark.parametrize("step_type", ["LabelEncoder", "OrdinalEncoder"])
def test_target_only_label_or_ordinal_encoding_before_split_is_allowed(step_type):
    """Label/Ordinal encoders with no `columns` selected only encode the
    target (y), which is standard leak-free practice - not a leakage risk."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("encode_target", step_type, ["load"], params={}),
        _node("split", "TrainTestSplitter", ["encode_target"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


@pytest.mark.parametrize("step_type", ["LabelEncoder", "OrdinalEncoder"])
def test_feature_column_label_or_ordinal_encoding_before_split_still_raises(step_type):
    """The same encoder types, configured with explicit feature `columns`,
    still fit on feature statistics - the leakage risk is unchanged."""
    nodes = [
        _node("load", "DataLoader", []),
        _node(
            "encode_features",
            step_type,
            ["load"],
            params={"columns": ["city", "country"]},
        ),
        _node("split", "TrainTestSplitter", ["encode_features"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


@pytest.mark.parametrize("step_type", ["LabelEncoder", "OrdinalEncoder"])
def test_explicit_target_column_selection_before_split_is_allowed(step_type):
    """Users commonly pick the target column explicitly from the column
    picker (columns == [target_column]) instead of leaving `columns` empty -
    this is still target-only encoding, not a leakage risk."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("encode_target", step_type, ["load"], params={"columns": ["species"]}),
        _node(
            "split",
            "TrainTestSplitter",
            ["encode_target"],
            params={"target_column": "species"},
        ),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


@pytest.mark.parametrize("step_type", ["LabelEncoder", "OrdinalEncoder"])
def test_target_plus_feature_columns_before_split_still_raises(step_type):
    """Mixing the target column with real feature columns still fits feature
    statistics on the whole dataset - the leakage risk remains."""
    nodes = [
        _node(
            "load",
            "DataLoader",
            [],
        ),
        _node(
            "encode_mixed",
            step_type,
            ["load"],
            params={"columns": ["species", "city"]},
        ),
        _node(
            "split",
            "TrainTestSplitter",
            ["encode_mixed"],
            params={"target_column": "species"},
        ),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_explicit_column_drop_before_split_is_allowed():
    """Dropping explicitly named columns is a fixed user decision (no
    learned statistic), so it may run before the train/test split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node(
            "drop_id",
            "DropMissingColumns",
            ["load"],
            params={"columns": ["passenger_id"], "missing_threshold": 0},
        ),
        _node("split", "TrainTestSplitter", ["drop_id"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_threshold_based_column_drop_before_split_still_raises():
    """With a positive missing-% threshold the node learns WHICH columns to
    drop from the fitted rows — that decision must stay after the split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node(
            "drop_sparse",
            "DropMissingColumns",
            ["load"],
            params={"missing_threshold": 50},
        ),
        _node("split", "TrainTestSplitter", ["drop_sparse"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_constant_imputation_before_split_is_allowed():
    """strategy='constant' fills with a user-fixed value — nothing is
    learned from the rows, so it may run before the train/test split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node(
            "fill",
            "SimpleImputer",
            ["load"],
            params={"strategy": "constant", "fill_value": 0, "columns": ["age"]},
        ),
        _node("split", "TrainTestSplitter", ["fill"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_statistic_imputation_before_split_still_raises():
    """mean/median/most_frequent learn from the fitted rows."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("fill", "SimpleImputer", ["load"], params={"strategy": "median"}),
        _node("split", "TrainTestSplitter", ["fill"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_explicit_missing_indicator_before_split_is_allowed():
    """Flagging explicitly named columns for missingness learns nothing
    from the rows, so it may run before the train/test split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node(
            "flags",
            "MissingIndicator",
            ["load"],
            params={"columns": ["age", "fare"]},
        ),
        _node("split", "TrainTestSplitter", ["flags"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_auto_detected_missing_indicator_before_split_still_raises():
    """With no explicit column list the node discovers WHICH columns contain
    missing values from the fitted rows — must stay after the split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("flags", "MissingIndicator", ["load"], params={}),
        _node("split", "TrainTestSplitter", ["flags"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_explicit_hash_encoding_before_split_is_allowed():
    """HashEncoder with a user-chosen column list learns nothing (deterministic
    hashing), so it may run before the train/test split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("hash", "HashEncoder", ["load"], params={"columns": ["city"], "n_features": 8}),
        _node("split", "TrainTestSplitter", ["hash"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_auto_detected_hash_encoding_before_split_still_raises():
    """With no columns key the node auto-detects WHICH columns are categorical
    from the fitted rows — must stay after the split."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("hash", "HashEncoder", ["load"], params={"n_features": 8}),
        _node("split", "TrainTestSplitter", ["hash"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)
