"""Tests for the standalone preprocessing leakage-safety diagnostic."""

import pytest

import skyulf
from skyulf.pipeline import SkyulfPipeline


def test_validate_leakage_safety_reports_learned_step_before_splitter():
    """A learned preprocessing step before a split should be reported."""
    config = {
        "preprocessing": [
            {"name": "fill missing", "transformer": "SimpleImputer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    warnings = skyulf.validate_leakage_safety(config, on_leakage="warn")

    assert len(warnings) == 1
    assert "Step 0 ('SimpleImputer')" in warnings[0]
    assert "step 1, 'TrainTestSplitter'" in warnings[0]


def test_validate_leakage_safety_raises_on_learned_step_before_splitter_by_default():
    """Since the enforcement batch the default on_leakage is 'raise'."""
    config = {
        "preprocessing": [
            {"name": "fill missing", "transformer": "SimpleImputer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    with pytest.raises(ValueError, match="SimpleImputer"):
        skyulf.validate_leakage_safety(config)


def test_validate_leakage_safety_allows_splitter_first():
    """A safe order should produce no warnings."""
    splitter_first = {
        "preprocessing": [
            {"name": "split", "transformer": "Split", "params": {}},
            {"name": "scale", "transformer": "StandardScaler", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(splitter_first) == []


def test_validate_leakage_safety_reports_missing_splitter():
    """No train/test boundary now yields an explicit advisory diagnostic."""
    no_splitter = {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {}},
        ],
        "modeling": {},
    }

    warnings = skyulf.validate_leakage_safety(no_splitter)

    assert len(warnings) == 1
    assert "No train/test split" in warnings[0]


def test_pipeline_validate_leakage_safety_delegates_to_module_function():
    """The pipeline convenience method should return the module result."""
    config = {
        "preprocessing": [
            {"name": "encode", "transformer": "OneHotEncoder", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert SkyulfPipeline(config).validate_leakage_safety(
        on_leakage="warn"
    ) == skyulf.validate_leakage_safety(config, on_leakage="warn")


def test_validate_leakage_safety_is_importable_from_package_top_level():
    """The diagnostic should be available from the package root."""
    from skyulf.leakage import validate_leakage_safety

    assert skyulf.validate_leakage_safety is validate_leakage_safety


def test_explicit_column_drop_before_splitter_is_allowed():
    """Dropping *named* columns is a user decision, not a learned statistic —
    safe before the split (edge case raised in review: 'directly dropping a
    column to not include it in the model')."""
    config = {
        "preprocessing": [
            {
                "name": "drop id",
                "transformer": "DropMissingColumns",
                "params": {"columns": ["passenger_id"], "missing_threshold": 0},
            },
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(config, on_leakage="warn") == []


def test_threshold_based_column_drop_before_splitter_still_raises():
    """The same node with a positive missing-% threshold *learns* which
    columns to drop from the fitted rows — that decision must stay
    post-split."""
    config = {
        "preprocessing": [
            {
                "name": "drop sparse",
                "transformer": "DropMissingColumns",
                "params": {"missing_threshold": 50},
            },
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    with pytest.raises(ValueError, match="DropMissingColumns"):
        skyulf.validate_leakage_safety(config)


def test_non_numeric_threshold_column_drop_before_splitter_is_allowed():
    """An unparseable ``missing_threshold`` (e.g. a stray string from a
    hand-edited pipeline JSON) cannot act as a learned threshold, so the
    node degrades to its explicit/no-op mode and stays allowed before the
    split — matching the node's own "non-positive/non-numeric threshold is
    not configured" handling."""
    config = {
        "preprocessing": [
            {
                "name": "drop id",
                "transformer": "DropMissingColumns",
                "params": {"columns": ["passenger_id"], "missing_threshold": "not-a-number"},
            },
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(config, on_leakage="warn") == []


# ---------------------------------------------------------------------------
# Pre-fit advisory warnings (surface the gate before model training)
# ---------------------------------------------------------------------------


def _tiny_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_fit_warns_about_learned_step_before_splitter(caplog):
    """fit() surfaces the leakage verdict as a warning before training."""
    import logging

    config = {
        "preprocessing": [
            {"name": "fill missing", "transformer": "SimpleImputer", "params": {}},
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": 0.3, "random_state": 42},
            },
        ],
        "modeling": {},
    }

    with caplog.at_level(logging.WARNING, logger="skyulf.pipeline"):
        SkyulfPipeline(config).fit(_tiny_frame(), target_column="target")

    assert any("before the train/test split" in r.message for r in caplog.records)


def test_fit_warns_when_no_split_is_defined(caplog):
    """A flat fit on an unsplit frame gets the advisory no-split diagnostic."""
    import logging

    config = {
        "preprocessing": [
            {"name": "fill missing", "transformer": "SimpleImputer", "params": {}},
        ],
        "modeling": {},
    }

    with caplog.at_level(logging.WARNING, logger="skyulf.pipeline"):
        SkyulfPipeline(config).fit(_tiny_frame(), target_column="target")

    assert any("No train/test split" in r.message for r in caplog.records)


def test_fit_stays_silent_for_externally_split_datasets(caplog):
    """When the caller supplies a SplitDataset the train/test boundary is
    provided externally and enforced by construction — no advisory noise."""
    import logging

    from skyulf.data.dataset import SplitDataset

    config = {
        "preprocessing": [
            {"name": "fill missing", "transformer": "SimpleImputer", "params": {}},
        ],
        "modeling": {},
    }
    df = _tiny_frame()
    dataset = SplitDataset(train=df.iloc[:7], test=df.iloc[7:])

    with caplog.at_level(logging.WARNING, logger="skyulf.pipeline"):
        SkyulfPipeline(config).fit(dataset, target_column="target")

    assert not any("leakage" in r.message.lower() for r in caplog.records)
    assert not any("No train/test split" in r.message for r in caplog.records)


def test_constant_imputation_before_splitter_is_allowed():
    """strategy='constant' fills with a user-fixed value — nothing is
    learned from the rows, so it may run before the split."""
    config = {
        "preprocessing": [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"strategy": "constant", "fill_value": 0, "columns": ["a"]},
            },
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(config, on_leakage="warn") == []


def test_statistic_imputation_before_splitter_still_raises():
    """mean/median/most_frequent strategies learn from the fitted rows."""
    for strategy in ("mean", "median", "most_frequent"):
        config = {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"strategy": strategy},
                },
                {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
            ],
            "modeling": {},
        }
        with pytest.raises(ValueError, match="SimpleImputer"):
            skyulf.validate_leakage_safety(config)


def test_explicit_missing_indicator_before_splitter_is_allowed():
    """Flagging *named* columns for missingness learns nothing from the rows —
    the column list comes from the config, so it may run before the split."""
    config = {
        "preprocessing": [
            {
                "name": "flags",
                "transformer": "MissingIndicator",
                "params": {"columns": ["age", "fare"]},
            },
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(config, on_leakage="warn") == []


def test_auto_detected_missing_indicator_before_splitter_still_raises():
    """With no explicit column list the node discovers WHICH columns contain
    missing values from the fitted rows — that decision must stay post-split."""
    config = {
        "preprocessing": [
            {"name": "flags", "transformer": "MissingIndicator", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    with pytest.raises(ValueError, match="MissingIndicator"):
        skyulf.validate_leakage_safety(config)


def test_explicit_hash_encoding_before_splitter_is_allowed():
    """HashEncoder with a user-chosen column list learns nothing: hashing is
    deterministic and fit() only records the config's columns/n_features."""
    config = {
        "preprocessing": [
            {
                "name": "hash",
                "transformer": "HashEncoder",
                "params": {"columns": ["city"], "n_features": 8},
            },
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(config, on_leakage="warn") == []


def test_explicit_empty_column_hash_encoding_before_splitter_is_allowed():
    """`columns: []` is the UI's 'nothing selected' no-op (fit returns {}),
    so it learns nothing either."""
    config = {
        "preprocessing": [
            {"name": "hash", "transformer": "HashEncoder", "params": {"columns": []}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    assert skyulf.validate_leakage_safety(config, on_leakage="warn") == []


def test_auto_detected_hash_encoding_before_splitter_still_raises():
    """With no columns key the node auto-detects WHICH columns are
    categorical from the fitted rows — that decision must stay post-split."""
    config = {
        "preprocessing": [
            {"name": "hash", "transformer": "HashEncoder", "params": {"n_features": 8}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ],
        "modeling": {},
    }

    with pytest.raises(ValueError, match="HashEncoder"):
        skyulf.validate_leakage_safety(config)
