"""Separate outcomes must remain pinned, keyed and subject to availability cutoffs."""

import os
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec


def _spec(**changes):
    """Keep ordinary date-free training while requiring timely external outcomes."""
    values: dict[str, Any] = {
        "table": "cat.data.features",
        "version": 0,
        "record_key_columns": ("id",),
        "input_columns": ("x",),
        "target_column": "y",
        "max_rows": 100,
        "max_bytes": 100000,
    }
    values.update(changes)
    return TrainingSpec(**values)


def _labeled_spec(**changes):
    """Provide a complete resolved outcome binding without touching a workspace."""
    values: dict[str, Any] = {
        "training_label_table": "cat.data.labels",
        "training_label_version": 2,
        "training_label_table_id": "labels-delta-id",
        "filter_unavailable_results": True,
        "result_available_at_column": "known_at",
        "result_cutoff": datetime(2026, 10, 17, tzinfo=UTC),
    }
    return _spec(**(values | changes))


def test_external_label_binding_changes_dataset_identity_and_roundtrips():
    """A changed outcome snapshot must not reuse the previous training dataset identity."""
    from skyulf.integrations.databricks.training.fitting.candidate import training_spec_payload

    spec = _labeled_spec()
    restored = TrainingSpec.from_payload(training_spec_payload(spec, "pandas"))
    assert restored == spec
    assert replace(spec, training_label_version=3).dataset_id != spec.dataset_id
    assert replace(spec, training_label_table_id="replaced-table").dataset_id != spec.dataset_id


@pytest.mark.parametrize(
    "changes",
    [
        {"training_label_table": "cat.data.features"},
        {"training_label_version": True},
        {"training_label_version": -1},
        {"training_label_table_id": ""},
        {"training_label_table": None},
    ],
)
def test_invalid_label_bindings_fail_before_read(changes):
    """Ambiguous versions and orphaned metadata must never reach Spark."""
    with pytest.raises(ValueError, match="training_label"):
        _labeled_spec(**changes)


def test_external_labels_require_explicit_availability_policy():
    """A separate outcome source cannot silently include future-known labels."""
    with pytest.raises(ValueError, match="availability"):
        _spec(training_label_table="cat.data.labels")


def test_unbound_spec_keeps_historical_identity_keys():
    """Opting out must not change old saved dataset fingerprints."""
    assert not any(key.startswith("training_label") for key in _spec()._identity_settings())


def test_prepared_frame_cannot_claim_another_outcome_snapshot():
    """Prepared candidate data must carry the exact binding used by its saved spec."""
    import pandas as pd

    from skyulf.integrations.databricks.data.training.training_labels import validate_label_binding

    frame = pd.DataFrame({"id": [1], "x": [2.0], "y": [3.0]})
    with pytest.raises(ValueError, match="binding"):
        validate_label_binding(frame, _labeled_spec())


@pytest.fixture
def outcome_tables(delta_spark):
    """Own two isolated Delta tables so native validation never rewrites example data."""
    namespace = os.environ.get("SKYULF_TEST_SCHEMA", "spark_catalog.default")
    names = [f"{namespace}.outcome_join_{uuid4().hex}" for _ in range(2)]
    try:
        yield names
    finally:
        for name in names:
            delta_spark.sql(f"DROP TABLE IF EXISTS {name}")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_delta_outcomes_are_pinned_and_late_labels_become_new_training_rows(
    delta_spark, outcome_tables, engine, tmp_path, monkeypatch
):
    """An unchanged feature snapshot must still detect newly available actual outcomes."""
    from skyulf.integrations.databricks.data.training.training_labels import pin_training_labels
    from skyulf.integrations.databricks.training.fitting.candidate import (
        read_training_snapshot,
        split_labeled_snapshot,
    )

    source, labels = outcome_tables
    known = datetime(2026, 10, 16, tzinfo=UTC)
    late = datetime(2026, 10, 18, tzinfo=UTC)
    delta_spark.createDataFrame(
        [(i, float(i)) for i in range(12)], "id long, x double"
    ).write.format("delta").saveAsTable(source)
    delta_spark.createDataFrame(
        [(i, float(i * 2), known if i < 10 else late) for i in range(11)],
        "id long, y double, known_at timestamp",
    ).write.format("delta").saveAsTable(labels)
    spec = pin_training_labels(
        delta_spark,
        _labeled_spec(
            table=source,
            training_label_table=labels,
            training_label_version=None,
            training_label_table_id=None,
        ),
    )
    original = read_training_snapshot(delta_spark, spec)
    train, holdout, excluded = split_labeled_snapshot(original, spec, engine=engine)
    assert len(train) + len(holdout) == 10 and excluded == 2
    delta_spark.createDataFrame(
        [(11, 22.0, known)], "id long, y double, known_at timestamp"
    ).write.format("delta").mode("append").saveAsTable(labels)
    assert read_training_snapshot(delta_spark, spec).equals(original)
    current = pin_training_labels(
        delta_spark, replace(spec, training_label_version=None, training_label_table_id=None)
    )
    fresh = read_training_snapshot(delta_spark, current)
    train, holdout, excluded = split_labeled_snapshot(fresh, current, engine=engine)
    assert len(train) + len(holdout) == 11 and excluded == 1
    assert current.dataset_id != spec.dataset_id
    assert fresh.attrs["training_labels"]["training_label_version"] == 1
    from skyulf.integrations.databricks.data.training.spark_retraining_data import (
        _eligible,
        _read_source,
    )

    assert _eligible(_read_source(delta_spark, current), current).count() == 11
    _fit_reload(delta_spark, current, engine, tmp_path, monkeypatch)


def _fit_reload(spark, spec, engine, tmp_path, monkeypatch):
    """Fit an actual candidate and replay its saved pipeline using the joined targets."""
    import numpy as np
    import pandas as pd
    import polars as pl

    from skyulf.inference.fitted_pipeline import load_pipeline
    from skyulf.integrations.databricks.training.fitting.candidate import fit_candidate
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec
    from skyulf.integrations.mlflow.runs.tracking import TrackingConfig, track_run

    monkeypatch.chdir(tmp_path)
    config = {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}
        ],
        "modeling": {"type": "linear_regression"},
    }
    tracking = TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{tmp_path / 'runs.db'}",
        experiment_name="outcome-join",
    )
    artifact_path = tmp_path / "candidate"
    with track_run(tracking, run_name=engine) as run:
        fitted = fit_candidate(
            spark,
            spec,
            config,
            run=run,
            pipeline_config=config,
            artifact_path=artifact_path,
            engine=engine,
            cv=CVSpec(),
            risk_category=None,
        )
    frame = pd.DataFrame({"x": [1.5, 3.5]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    expected = fitted.artifact.pipeline.predict(native)
    restored = load_pipeline(artifact_path)
    np.testing.assert_allclose(restored.pipeline.predict(native), expected, atol=1e-10)
    np.testing.assert_allclose(expected, [3.0, 7.0], atol=1e-10)
    assert fitted.training_rows + fitted.holdout_rows == 11
    assert fitted.spec.training_label_version == 1


@pytest.mark.parametrize(
    "bad_keys,expected", [([1, 1], "duplicate record keys"), ([1, None], "null or nonfinite")]
)
def test_delta_outcomes_reject_ambiguous_keys(delta_spark, outcome_tables, bad_keys, expected):
    """A many-to-many or missing-key join must fail before materializing training rows."""
    from skyulf.integrations.databricks.data.training.training_labels import pin_training_labels
    from skyulf.integrations.databricks.training.fitting.candidate import read_training_snapshot

    source, labels = outcome_tables
    delta_spark.createDataFrame([(1, 2.0)], "id long, x double").write.format("delta").saveAsTable(
        source
    )
    delta_spark.createDataFrame(
        [(key, 3.0, datetime(2026, 10, 16, tzinfo=UTC)) for key in bad_keys],
        "id long, y double, known_at timestamp",
    ).write.format("delta").saveAsTable(labels)
    spec = pin_training_labels(
        delta_spark,
        _labeled_spec(
            table=source,
            training_label_table=labels,
            training_label_version=None,
            training_label_table_id=None,
        ),
    )
    with pytest.raises(ValueError, match=expected):
        read_training_snapshot(delta_spark, spec)


def test_delta_branch_plan_replays_shared_outcomes(delta_spark, outcome_tables, workflow_config):
    """A saved multi-target plan must preserve both targets after the outcome table advances."""
    from skyulf.integrations.databricks.training.branches import (
        branch_training_payload,
        prepare_training_branches,
        restore_training_branches,
    )
    from skyulf.integrations.databricks.training.fitting.candidate import read_training_snapshot

    source, labels = outcome_tables
    delta_spark.createDataFrame([(1, 2.0)], "id long, x double").write.format("delta").saveAsTable(
        source
    )
    delta_spark.createDataFrame(
        [(1, 4.0, 6.0, datetime(2026, 10, 16, tzinfo=UTC))],
        "id long, y double, z double, known_at timestamp",
    ).write.format("delta").saveAsTable(labels)
    configs = {}
    for target in ("y", "z"):
        configs[target] = {
            **workflow_config,
            "engine": "pandas",
            "training_table": source,
            "training_version": 0,
            "training_label_table": labels,
            "training_window_mode": "full_snapshot",
            "split_strategy": "random",
            "score_handoff": "disabled",
            "record_key_columns": ["id"],
            "target_column": target,
            "model_name": f"workspace.test.{target}",
            "result_available_at_column": "known_at",
            "result_cutoff": "2026-10-17T00:00:00+00:00",
            **dict.fromkeys(
                (
                    "start",
                    "holdout_start",
                    "cutoff",
                    "event_column",
                    "window_timezone",
                    "monthly_lookback_months",
                )
            ),
        }
    plan = prepare_training_branches(delta_spark, configs, champion_versions={"y": None, "z": None})
    delta_spark.sql(f"UPDATE {labels} SET y = 400.0, z = 600.0")
    restored = restore_training_branches(branch_training_payload(plan))
    assert restored == plan
    assert {branch.spec.training_label_version for branch in restored} == {0}
    for branch in restored:
        frame = read_training_snapshot(delta_spark, branch.spec)
        assert frame[branch.name].tolist() == [{"y": 4.0, "z": 6.0}[branch.name]]
