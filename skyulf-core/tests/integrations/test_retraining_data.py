"""Drift retraining requires changed eligible training values, not another split."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from skyulf.integrations.databricks import local_retraining, local_workflow
from skyulf.integrations.databricks.monitoring_config import MonitorConfig


@pytest.fixture
def freshness(monkeypatch):
    """Exercise the production split while replacing only cloud snapshot and registry reads."""
    from skyulf.integrations.databricks import retraining_data

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
        train, _, _ = local_retraining.split_labeled_snapshot(
            snapshot(None, spec), spec, engine=saved["engine"]
        )
        artifact = SimpleNamespace(manifest=SimpleNamespace(fitted_engine=saved["engine"]))
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
    from skyulf.integrations.databricks.retraining_data import assess_training_data

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
