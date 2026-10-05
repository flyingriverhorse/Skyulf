"""Explicit missing-label policy and early branch tracking for local candidates."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.training.fitting import local_retraining as retraining


def _spec(**changes):
    """Keep optional label and date policies inactive unless a test enables them."""
    values: dict[str, Any] = {
        "table": "workspace.test.labels",
        "version": 4,
        "record_key_columns": ("id",),
        "input_columns": ("x",),
        "target_column": "target",
        "max_rows": 100,
        "max_bytes": 100000,
    }
    return retraining.LocalTrainingSpec(**{**values, **changes})


def _frame():
    """Missing targets must be excluded without changing labeled feature values."""
    frame = pd.DataFrame({"id": range(20), "x": range(20), "target": range(20)})
    frame["target"] = frame["target"].astype(float)
    frame.loc[[0, 1], "target"] = float("nan")
    return frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_drop_missing_labels_produces_fit_ready_partitions(tmp_path, engine):
    """Both local engines must train only on known outcomes without imputation."""
    spec = _spec(drop_missing_labels=True)
    train, heldout, unavailable = retraining.split_labeled_snapshot(_frame(), spec, engine=engine)
    assert unavailable == 2
    assert set(train.x).union(heldout.x) == set(range(2, 20))
    assert not train.target.isna().any() and not heldout.target.isna().any()
    native = pl.from_pandas(train) if engine == "polars" else train
    artifact = retraining.fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / engine,
        max_rows=100,
        max_bytes=100000,
    )
    native_holdout = pl.from_pandas(heldout) if engine == "polars" else heldout
    metrics = retraining.evaluate_local_holdout(artifact, native_holdout, target_column="target")
    assert metrics["heldout_rmse"] == pytest.approx(0, abs=1e-8)


def test_missing_and_late_labels_are_counted_once():
    """Overlapping missing-target and time filters must count the union of excluded rows."""
    frame = _frame()
    frame["available"] = pd.Timestamp("2026-03-01", tz="UTC")
    frame.loc[[1, 2], "available"] = pd.Timestamp("2026-04-01", tz="UTC")
    frame.loc[3, "available"] = pd.NaT
    train, heldout, unavailable = retraining.split_labeled_snapshot(
        frame,
        _spec(
            drop_missing_labels=True,
            filter_unavailable_results=True,
            result_available_at_column="available",
            result_cutoff=datetime(2026, 3, 15, tzinfo=UTC),
        ),
    )
    assert unavailable == 4
    assert set(train.x).union(heldout.x) == set(range(4, 20))


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_missing_label_policy_requires_a_boolean(invalid):
    """Truthy configuration values must never silently enable target exclusion."""
    with pytest.raises(ValueError, match="drop_missing_labels must be boolean"):
        _spec(drop_missing_labels=invalid)


def test_default_policy_preserves_saved_identity_and_missing_target_error():
    """Old evidence must retain its dataset identity and strict target validation."""
    spec = _spec()
    assert spec.dataset_id == (
        "workspace.test.labels@4/random/"
        "6bc58cd1cb7f093da1979997e18737fe3dada247e718d2e4b5dd274320e2af27"
    )
    payload = retraining.training_spec_payload(spec, "pandas")
    payload.pop("drop_missing_labels", None)
    restored = retraining.LocalTrainingSpec.from_payload(payload)
    assert restored.drop_missing_labels is False
    assert replace(spec, drop_missing_labels=True).dataset_id != spec.dataset_id
    with pytest.raises(ValueError, match="nonnull targets"):
        retraining.split_labeled_snapshot(_frame(), restored)


def test_candidate_tags_survive_fit_failure(monkeypatch, tmp_path):
    """A failed child fit must retain its parent and branch identifiers in MLflow."""
    run = Mock(run_id="child-run")
    tags = {"mlflow.parentRunId": "parent-run", "skyulf.branch": "supervised"}
    observed = {}
    run.set_tags.side_effect = observed.update

    @contextmanager
    def tracked(*args, **kwargs):
        """Supply a tracked run without requiring an external tracking server."""
        yield run

    def fail_fit(*args, **kwargs):
        """Check durable lineage is installed before the operation that can fail."""
        assert observed == tags
        raise RuntimeError("fit failed")

    monkeypatch.setattr(retraining, "track_run", tracked)
    monkeypatch.setattr(retraining, "fit_candidate", fail_fit)
    with pytest.raises(RuntimeError, match="fit failed"):
        retraining.train_local_candidate(
            None,
            _spec(),
            {"preprocessing": [], "modeling": {"type": "linear_regression"}},
            model_name="unused",
            tracking_uri="unused",
            registry_uri="unused",
            experiment_name="unused",
            run_name="unused",
            artifact_path=tmp_path,
            metric="heldout_rmse",
            min_improvement=0,
            run_tags=tags,
        )


@pytest.mark.parametrize("filter_times", [False, True])
def test_spark_sampling_excludes_missing_targets_before_validation(delta_spark, filter_times):
    """Spark sampling must exclude null and NaN labels before validating or sampling targets."""
    frame = delta_spark.createDataFrame(
        [
            (i, float(i), None if i == 0 else float("nan") if i == 1 else float(i), i)
            for i in range(20)
        ],
        "id long, x double, target double, available long",
    )
    settings = (
        {
            "filter_unavailable_results": True,
            "result_available_at_column": "available",
            "result_cutoff": datetime(1970, 1, 1, 0, 0, 0, 17, tzinfo=UTC),
        }
        if filter_times
        else {}
    )
    spec = _spec(drop_missing_labels=True, training_sample_rows=8, **settings)
    sampled, evidence = retraining._sample_training_source(frame, spec)
    local = sampled.toPandas()
    local.attrs["training_selection"] = evidence
    if filter_times:
        local["available"] = pd.to_datetime(local["available"], unit="us", utc=True)
    train, heldout, unavailable = retraining.split_labeled_snapshot(local, spec)
    assert len(train) + len(heldout) == 8
    assert set(local.id).issubset(set(range(2, 18 if filter_times else 20)))
    assert evidence["source_rows"] == 20
    assert unavailable == evidence["unavailable_labels"] == (4 if filter_times else 2)
