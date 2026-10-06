"""Drift retraining requires changed eligible training values, not another split."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from skyulf.integrations.databricks.lifecycle import local_workflow
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.training.fitting import local_retraining


@pytest.fixture
def freshness(monkeypatch):
    """Exercise the production split while replacing only cloud snapshot and registry reads."""
    from skyulf.integrations.databricks.data.training import retraining_data

    now = datetime(2026, 10, 1, tzinfo=UTC)
    config = {
        "training_table": "workspace.demo.training",
        "training_version": None,
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "y",
        "max_rows": 1000,
        "max_input_mb": 1,
        "engine": "pandas",
        "pipeline": {},
    }
    old = pd.DataFrame({"id": range(20), "x": range(20), "y": range(20)})
    state: dict[str, Any] = {"old": old, "current": old.copy(), "config": config, "now": now}

    def snapshot(spark, spec):
        """Represent exact Delta snapshots and Spark-side temporal selection."""
        frame = state["old" if spec.version == 0 else "current"].copy()
        if spec.event_column:
            values = pd.to_datetime(frame[spec.event_column], utc=True)
            frame = frame.loc[(values >= spec.start) & (values < spec.cutoff)]
        return frame.reset_index(drop=True)

    def reference(spark, monitor, **kwargs):
        """Provide the same saved split that the registry evidence would verify."""
        saved = state.get("saved_config", state["config"])
        spec = local_workflow.training_spec(
            local_workflow.training_settings({**saved, "training_version": 0}, now)
        )
        spec = replace(spec, **state.get("saved_spec_overrides", {}))
        train, _, _ = local_retraining.split_labeled_snapshot(
            snapshot(None, spec), spec, engine=saved["engine"]
        )
        artifact = SimpleNamespace(
            manifest=SimpleNamespace(fitted_engine=saved["engine"]),
            pipeline=SimpleNamespace(config={}),
        )
        return artifact, spec, train, {"model_version": "1"}

    monkeypatch.setattr(local_retraining, "read_training_snapshot", snapshot)
    monkeypatch.setattr(retraining_data, "load_monitoring_reference", reference)
    history = SimpleNamespace(
        select=lambda *args: SimpleNamespace(
            orderBy=lambda *args, **kwargs: SimpleNamespace(first=lambda: {"version": 1})
        )
    )
    state["spark"] = SimpleNamespace(sql=lambda query: history)
    state["monitor"] = MonitorConfig(
        environment="test",
        project="demo",
        model_name="workspace.demo.model",
        model_version="1",
        source_table="workspace.demo.score",
        prediction_table="workspace.demo.predictions",
    )
    return state


def _assess(state):
    """Run the public assessment with the fixture's current source and clock."""
    from skyulf.integrations.databricks.data.training.retraining_data import assess_training_data

    return assess_training_data(state["spark"], state["monitor"], state["config"], state["now"])


def test_metadata_commit_and_row_order_do_not_make_data_fresh(freshness):
    """A newer Delta version alone must never submit redundant training."""
    initial = _assess(freshness)
    freshness["current"] = freshness["current"].iloc[::-1].copy()
    repeated = _assess(freshness)
    assert repeated["status"] == "no_new_training_data"
    assert repeated["changed_rows"] == 0
    assert repeated["content_sha256"] == initial["content_sha256"]
    assert repeated["source_version"] == 1


def test_random_repartitioning_old_holdout_is_not_new_data(freshness):
    """A different random seed must not reclassify previously seen holdout as fresh data."""
    freshness["saved_config"] = freshness["config"].copy()
    freshness["config"]["random_state"] = 99
    result = _assess(freshness)
    assert result["status"] == "no_new_training_data"
    assert result["changed_rows"] == 0


def test_new_labeled_random_rows_count_only_training_members(freshness):
    """New holdout rows must not inflate the amount of fresh data used for fitting."""
    freshness["current"] = pd.DataFrame({"id": range(40), "x": range(40), "y": range(40)})
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert 0 < result["changed_rows"] < 20
    assert result["training_rows"] == 32


def test_changed_target_counts_as_fresh_training_data(freshness):
    """Corrections to existing labels can justify fitting without new record identities."""
    freshness["current"]["y"] += 100
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert result["changed_rows"] == result["training_rows"] == 16


def test_future_and_missing_results_do_not_count(freshness):
    """Unavailable targets must be excluded by the same policy used by training."""
    freshness["config"].update(
        filter_unavailable_results=True, result_available_at_column="available_at"
    )
    freshness["old"]["available_at"] = freshness["now"] - timedelta(days=1)
    extra = pd.DataFrame(
        {
            "id": [20, 21],
            "x": [20, 21],
            "y": [20, None],
            "available_at": [freshness["now"] + timedelta(days=1), None],
        }
    )
    freshness["current"] = pd.concat([freshness["old"], extra], ignore_index=True)
    result = _assess(freshness)
    assert result["status"] == "no_new_training_data"
    assert result["training_rows"] == 16


def test_temporal_requires_rows_to_reach_training_window(freshness):
    """Appending within holdout is insufficient until the moving boundary admits training rows."""
    now = freshness["now"]
    freshness["config"].update(
        training_window_mode="rolling_days",
        split_strategy="temporal",
        lookback_days=90,
        holdout_days=10,
        event_column="event_time",
        test_size=None,
        random_state=None,
        stratify=None,
    )
    freshness["old"]["event_time"] = [now - timedelta(days=30 - i) for i in range(20)]
    extra = pd.DataFrame(
        {
            "id": [20, 21, 22, 23],
            "x": [20, 21, 22, 23],
            "y": [20, 21, 22, 23],
            "event_time": [now - timedelta(days=i) for i in (9, 8, 2, 1)],
        }
    )
    # Keep an existing holdout in the saved model's snapshot.
    freshness["old"].loc[18:, "event_time"] = [now - timedelta(days=10), now - timedelta(days=9)]
    freshness["current"] = pd.concat([freshness["old"], extra], ignore_index=True)
    unchanged = _assess(freshness)
    freshness["now"] += timedelta(days=5)
    changed = _assess(freshness)
    assert unchanged["status"] == "no_new_training_data"
    assert changed["status"] == "ready"
    assert changed["changed_rows"] == 4


def test_pinned_source_is_rejected_before_cloud_reads(freshness):
    """Automatic retraining must never silently keep fitting a pinned old snapshot."""
    freshness["config"]["training_version"] = 0
    with pytest.raises(ValueError, match="training_version"):
        _assess(freshness)


def test_incompatible_source_contract_is_rejected(freshness):
    """Changing feature semantics requires explicit training rather than comparing unrelated rows."""
    freshness["saved_config"] = freshness["config"].copy()
    freshness["config"]["training_table"] = "workspace.other.training"
    with pytest.raises(ValueError, match="training source"):
        _assess(freshness)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("available", [False, True])
def test_multi_target_freshness_preserves_branch_label_eligibility(freshness, engine, available):
    """Partial labels must follow the same per-target population during training and retraining."""
    freshness["config"].update(training_layout="multi_target", engine=engine)
    freshness["saved_spec_overrides"] = {"drop_missing_labels": True}
    freshness["old"].loc[:3, "y"] = None
    extra = pd.DataFrame(
        {"id": range(20, 40), "x": range(20, 40), "y": range(20, 40) if available else [None] * 20}
    )
    freshness["current"] = pd.concat([freshness["old"], extra], ignore_index=True)

    result = _assess(freshness)

    assert result["status"] == ("ready" if available else "no_new_training_data")
    if available:
        assert 0 < result["changed_rows"] < 20
    else:
        assert result["changed_rows"] == 0
        assert result["training_rows"] == 12


def test_changed_missing_label_policy_requires_explicit_training(freshness):
    """Automatic freshness comparisons must reject a different label eligibility contract."""
    freshness["saved_spec_overrides"] = {"drop_missing_labels": True}
    with pytest.raises(ValueError, match="training source and split contract"):
        _assess(freshness)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pre_split_filters_define_freshness_population(freshness, engine):
    """Rows excluded by the real engine-specific pre-split recipe cannot trigger a fit."""
    freshness["config"].update(
        engine=engine,
        pre_split_steps=[
            {
                "name": "valid_features",
                "transformer": "DropMissingRows",
                "params": {"subset": ["x"]},
            }
        ],
    )
    extra = pd.DataFrame({"id": [20, 21], "x": [None, None], "y": [20, 21]})
    freshness["current"] = pd.concat([freshness["old"], extra], ignore_index=True)
    result = _assess(freshness)
    assert result["status"] == "no_new_training_data"
    assert result["training_rows"] == 16


def _weight_filtered_source(state, engine="pandas"):
    """Keep source values fixed while admitting previously excluded weight rows."""
    state["config"].update(
        engine=engine,
        weight_column="w",
        reserved_weight_columns=["w"],
        pre_split_steps=[
            {
                "name": "keep",
                "transformer": "ManualBounds",
                "params": {"bounds": {"w": {"lower": 5}}},
            }
        ],
    )
    state["old"]["w"] = [1.0] * 10 + [10.0] * 10
    state["current"]["w"] = 10.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_weight_filter_changes_do_not_make_training_data_fresh(freshness, engine):
    """Changing only filter eligibility through weights must not request an automatic fit."""
    _weight_filtered_source(freshness, engine)
    result = _assess(freshness)
    assert result["training_rows"] == 16
    assert result["status"] == "no_new_training_data"
    assert result["changed_rows"] == 0


@pytest.mark.parametrize("column", ["x", "y"])
def test_weight_filtered_source_still_detects_real_changes(freshness, column):
    """An expanded historical population cannot hide real feature or label corrections."""
    _weight_filtered_source(freshness)
    freshness["current"][column] += 100
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert result["changed_rows"] == result["training_rows"] == 16


def test_weight_filter_does_not_hide_new_training_rows(freshness):
    """New observations remain fresh only when they enter the actual current fit partition."""
    _weight_filtered_source(freshness)
    freshness["current"] = pd.DataFrame(
        {"id": range(40), "x": range(40), "y": range(40), "w": [10.0] * 40}
    )
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert 0 < result["changed_rows"] < 20
    assert result["training_rows"] == 32


@pytest.mark.parametrize("eligibility", ["missing", "late"])
def test_weight_filter_preserves_newly_available_labels(freshness, eligibility):
    """Historical rows without usable labels must not suppress genuinely new training data."""
    _weight_filtered_source(freshness)
    if eligibility == "missing":
        freshness["config"]["drop_missing_labels"] = True
        freshness["old"].loc[:9, "y"] = None
    else:
        freshness["config"].update(
            filter_unavailable_results=True, result_available_at_column="available_at"
        )
        freshness["old"]["available_at"] = freshness["now"] - timedelta(days=1)
        freshness["old"].loc[:9, "available_at"] = freshness["now"] + timedelta(days=1)
        freshness["current"]["available_at"] = freshness["now"] - timedelta(days=1)
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert result["changed_rows"] == 8


def test_weight_filter_temporal_baseline_allows_holdout_to_age_in(freshness):
    """Weight-only admissions are old data while genuinely aged-in holdout remains new."""
    _weight_filtered_source(freshness)
    now = freshness["now"]
    freshness["config"].update(
        training_window_mode="rolling_days",
        split_strategy="temporal",
        lookback_days=90,
        holdout_days=10,
        event_column="event_time",
        test_size=None,
        random_state=None,
        stratify=None,
    )
    freshness["old"]["event_time"] = [now - timedelta(days=30 - i) for i in range(20)]
    freshness["old"].loc[18:, "event_time"] = [now - timedelta(days=10), now - timedelta(days=9)]
    freshness["current"]["event_time"] = freshness["old"]["event_time"]
    extra = pd.DataFrame(
        {
            "id": [20, 21],
            "x": [20, 21],
            "y": [20, 21],
            "w": [10.0, 10.0],
            "event_time": [now - timedelta(days=2), now - timedelta(days=1)],
        }
    )
    freshness["current"] = pd.concat([freshness["current"], extra], ignore_index=True)
    unchanged = _assess(freshness)
    freshness["now"] += timedelta(days=5)
    changed = _assess(freshness)
    assert unchanged["status"] == "no_new_training_data"
    assert changed["status"] == "ready"
    assert changed["changed_rows"] == 2


def test_weight_filter_baseline_does_not_validate_excluded_fit_weights(freshness):
    """Historically filtered invalid weights must not be validated as new training inputs."""
    _weight_filtered_source(freshness)
    freshness["old"].loc[:9, "w"] = -1.0
    result = _assess(freshness)
    assert result["status"] == "no_new_training_data"
    assert result["changed_rows"] == 0


@pytest.mark.parametrize("changed_labels", [False, True])
def test_weight_filter_baseline_does_not_run_dedup_on_excluded_conflicts(freshness, changed_labels):
    """Changing a weight-selected duplicate must not invent new labels or break baseline replay."""
    _weight_filtered_source(freshness)
    freshness["config"]["pre_split_steps"].append(
        {"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["x"]}}
    )
    for frame in (freshness["old"], freshness["current"]):
        frame["x"] = list(range(10)) * 2
        frame["y"] = list(range(100, 110)) + list(range(10))
    freshness["current"]["w"] = [10.0] * 10 + [1.0] * 10
    if changed_labels:
        freshness["current"].loc[:9, "y"] += 100
    result = _assess(freshness)
    assert result["training_rows"] == 8
    assert result["changed_rows"] == (8 if changed_labels else 0)
    assert result["status"] == ("ready" if changed_labels else "no_new_training_data")


def test_new_weight_declaration_protects_previously_unweighted_filter(freshness):
    """Enabling weights must protect a formerly ordinary filter source column too."""
    _weight_filtered_source(freshness)
    freshness["saved_config"] = {
        **freshness["config"],
        "weight_column": None,
        "reserved_weight_columns": [],
    }
    result = _assess(freshness)
    assert result["status"] == "no_new_training_data"
    assert result["changed_rows"] == 0


def test_weight_filter_counterfactual_keeps_historical_label_alternatives(freshness):
    """Corrected labels stay fresh when old counterfactual duplicate groups disagree."""
    _weight_filtered_source(freshness)
    freshness["config"]["pre_split_steps"].append(
        {"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["x"]}}
    )
    for frame in (freshness["old"], freshness["current"]):
        frame["x"] = list(range(10)) * 2
        frame["y"] = list(range(100, 110)) + list(range(10))
    freshness["current"]["y"] = list(range(200, 210)) * 2
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert result["changed_rows"] == result["training_rows"] == 8


@pytest.mark.parametrize("survivors", [0, 1, 2, 3])
def test_weight_filter_allows_tiny_counterfactual_when_new_data_is_valid(freshness, survivors):
    """Historical comparison needs no minimum fit size when current new rows train normally."""
    _weight_filtered_source(freshness)
    freshness["current"]["w"] = 0.0
    freshness["current"].loc[: survivors - 1, "w"] = 10.0
    extra = pd.DataFrame({"id": range(20, 40), "x": range(20, 40), "y": range(20, 40), "w": 10.0})
    freshness["current"] = pd.concat([freshness["current"], extra], ignore_index=True)
    result = _assess(freshness)
    assert result["status"] == "ready"
    assert 0 < result["changed_rows"] <= 20
    assert result["changed_rows"] <= result["training_rows"]
