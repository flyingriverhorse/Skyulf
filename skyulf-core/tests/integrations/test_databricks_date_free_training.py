"""Date-free source selection, independent availability and reproducible holdouts."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks import local_retraining as retraining


def _spec(**changes):
    """Provide only ordinary labeled-table fields unless a test opts into dates."""
    settings: dict[str, Any] = {
        "table": "workspace.test.labels",
        "version": 4,
        "record_key_columns": ("tenant", "id"),
        "input_columns": ("x",),
        "target_column": "target",
        "max_rows": 100,
        "max_bytes": 100000,
    }
    settings.update(changes)
    return retraining.LocalTrainingSpec(**settings)


def _frame():
    """Composite identities and perfect linear targets make leakage and fit observable."""
    return pd.DataFrame(
        {
            "tenant": ["a"] * 10 + ["b"] * 10,
            "id": list(range(10)) * 2,
            "x": list(range(20)),
            "target": [2 * value + 1 for value in range(20)],
        }
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_date_free_split_fits_both_engines_and_ignores_source_order(tmp_path, engine):
    """Ordinary tables must train reproducibly without synthetic dates or identity features."""
    spec = _spec()
    train, heldout, excluded = retraining.split_labeled_snapshot(_frame(), spec)
    reordered = retraining.split_labeled_snapshot(_frame().sample(frac=1, random_state=5), spec)
    pd.testing.assert_frame_equal(train, reordered[0])
    pd.testing.assert_frame_equal(heldout, reordered[1])
    assert len(train) == 16 and len(heldout) == 4 and excluded == 0
    assert list(train.columns) == ["x", "target"]
    assert set(train.x).isdisjoint(heldout.x)
    native_train = pl.from_pandas(train) if engine == "polars" else train
    native_heldout = pl.from_pandas(heldout) if engine == "polars" else heldout
    artifact = retraining.fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=native_train, test=native_train.head(0)),
        target_column="target",
        artifact_path=tmp_path / engine,
        max_rows=100,
        max_bytes=100000,
    )
    metrics = retraining.evaluate_local_holdout(artifact, native_heldout, target_column="target")
    assert metrics["heldout_rmse"] == pytest.approx(0, abs=1e-8)


def test_random_availability_has_independent_cutoff_and_no_event_column():
    """Late and unknown outcomes are excluded even when no event date exists."""
    frame = _frame()
    frame["available"] = pd.Timestamp("2026-03-01", tz="UTC")
    frame.loc[0, "available"] = pd.NaT
    frame.loc[1, "available"] = pd.Timestamp("2026-04-01", tz="UTC")
    frame.loc[[0, 1], "target"] = float("nan")
    spec = _spec(
        filter_unavailable_results=True,
        result_available_at_column="available",
        result_cutoff=datetime(2026, 3, 15, tzinfo=UTC),
    )
    train, heldout, excluded = retraining.split_labeled_snapshot(frame, spec)
    assert excluded == 2 and len(train) + len(heldout) == 18
    assert set(train.x).union(heldout.x) == set(range(2, 20))
    frame.loc[1, "target"] = 3
    later = _spec(
        version=5,
        filter_unavailable_results=True,
        result_available_at_column="available",
        result_cutoff=datetime(2026, 4, 1, tzinfo=UTC),
    )
    assert retraining.split_labeled_snapshot(frame, later)[2] == 1
    assert later.dataset_id != spec.dataset_id


def test_explicit_stratification_never_silently_downgrades():
    """Rare classes must fail instead of letting Core silently disable requested stratification."""
    frame = _frame()
    frame["target"] = [0] * 10 + [1] * 10
    train, heldout, _ = retraining.split_labeled_snapshot(frame, _spec(stratify=True))
    assert train.target.value_counts().to_dict() == {0: 8, 1: 8}
    assert heldout.target.value_counts().to_dict() == {0: 2, 1: 2}
    frame.loc[0, "target"] = 2
    with pytest.raises(ValueError, match="stratif"):
        retraining.split_labeled_snapshot(frame, _spec(stratify=True))


def test_null_target_is_rejected_in_labeled_dataset_mode():
    """Disabling availability requires a fully labeled source, without filling targets."""
    frame = _frame()
    frame.loc[0, "target"] = float("nan")
    with pytest.raises(ValueError, match="nonnull targets"):
        retraining.split_labeled_snapshot(frame, _spec())


@pytest.mark.parametrize(
    "changes",
    [
        {"event_column": "event"},
        {"start": datetime(2026, 1, 1, tzinfo=UTC)},
        {"result_available_at_column": "available"},
        {"filter_unavailable_results": True},
        {"split_strategy": "temporal"},
        {"random_state": None},
        {"stratify": "yes"},
    ],
)
def test_inactive_or_incomplete_policies_are_rejected(changes):
    """Contradictory settings must not silently choose another training policy."""
    with pytest.raises((ValueError, TypeError)):
        _spec(**changes)


@pytest.mark.parametrize("field,value", [("random_state", 42.0), ("stratify", 0)])
def test_temporal_inactive_random_defaults_still_require_correct_types(field, value):
    """Inactive defaults must not let mistyped settings survive a later policy switch."""
    with pytest.raises(ValueError, match="random split settings"):
        _spec(
            split_strategy="temporal",
            event_column="event",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            holdout_start=datetime(2026, 2, 1, tzinfo=UTC),
            cutoff=datetime(2026, 3, 1, tzinfo=UTC),
            **{field: value},
        )


def test_direct_training_rejects_regression_stratification_before_read(monkeypatch, tmp_path):
    """Direct SDK calls must not stratify repeated regression targets as invented classes."""
    reader = Mock(side_effect=AssertionError("source opened"))
    monkeypatch.setattr(retraining, "read_training_snapshot", reader)
    with pytest.raises(ValueError, match="classification"):
        retraining.train_local_candidate(
            None,
            _spec(stratify=True),
            {"preprocessing": [], "modeling": {"type": "linear_regression"}},
            model_name="unused",
            tracking_uri="unused",
            registry_uri="unused",
            experiment_name="unused",
            run_name="unused",
            artifact_path=tmp_path,
            metric="heldout_rmse",
            min_improvement=0,
        )
    reader.assert_not_called()


def test_saved_membership_detects_changed_identities_but_replays_reordered_rows():
    """Approval must fail when replay changes the original final holdout identities."""
    spec = _spec()
    _, heldout, _ = retraining.split_labeled_snapshot(_frame(), spec)
    pinned = replace(spec, holdout_key_sha256=heldout.attrs["holdout_key_sha256"])
    replay = retraining.split_labeled_snapshot(_frame().iloc[::-1], pinned)[1]
    pd.testing.assert_frame_equal(heldout, replay)
    changed = _frame()
    changed["id"] += 100
    with pytest.raises(ValueError, match="membership"):
        retraining.split_labeled_snapshot(changed, pinned)
    assert replace(spec, random_state=43).dataset_id != spec.dataset_id
    assert replace(spec, max_rows=1000).dataset_id == spec.dataset_id
    assert (
        replace(spec, holdout_key_sha256="a" * 64).dataset_id
        != replace(spec, holdout_key_sha256="b" * 64).dataset_id
    )


def test_available_null_target_fails_and_temporal_cutoffs_are_independent():
    """Outcomes after the event window remain eligible only when available at result cutoff."""
    frame = _frame()
    frame["event"] = pd.to_datetime(["2026-01-15"] * 10 + ["2026-02-15"] * 10, utc=True)
    frame["available"] = pd.Timestamp("2026-03-10", tz="UTC")
    spec = _spec(
        split_strategy="temporal",
        event_column="event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 2, 1, tzinfo=UTC),
        cutoff=datetime(2026, 3, 1, tzinfo=UTC),
        filter_unavailable_results=True,
        result_available_at_column="available",
        result_cutoff=datetime(2026, 3, 15, tzinfo=UTC),
    )
    train, heldout, excluded = retraining.split_labeled_snapshot(frame, spec)
    assert len(train) == 10 and len(heldout) == 10 and excluded == 0
    unfiltered = replace(
        spec, filter_unavailable_results=False, result_available_at_column=None, result_cutoff=None
    )
    assert retraining.split_labeled_snapshot(frame.drop(columns="available"), unfiltered)[2] == 0
    frame.loc[0, "target"] = float("nan")
    with pytest.raises(ValueError, match="nonnull targets"):
        retraining.split_labeled_snapshot(frame, spec)


def test_date_free_monthly_pins_full_latest_snapshot_and_invocation_result_cutoff():
    """Monthly execution dates must not create an implicit event window for random training."""
    from skyulf.integrations.databricks.local_workflow import _monthly_training_spec

    config = {
        "training_table": "workspace.test.labels",
        "record_key_columns": ["tenant", "id"],
        "input_columns": ["x"],
        "target_column": "target",
        "max_rows": 100,
        "max_input_mb": 1,
    }
    spark = Mock()
    spark.sql.return_value.select.return_value.orderBy.return_value.first.return_value = {
        "version": 8
    }
    now = datetime(2026, 9, 25, 12, 34, tzinfo=UTC)
    spec = _monthly_training_spec(spark, config, now)
    assert spec.version == 8 and spec.start is None and spec.event_column is None
    config.update(filter_unavailable_results=True, result_available_at_column="available")
    spec = _monthly_training_spec(spark, config, now)
    assert spec.result_cutoff == now and spec.cutoff is None
    with pytest.raises(ValueError, match="monthly_lookback_months"):
        _monthly_training_spec(spark, {**config, "monthly_lookback_months": 4}, now)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_random_candidate_approval_replays_saved_membership_after_config_changes(
    monkeypatch,
    tmp_path,
    engine,
):
    """Persisted MLflow evidence must preserve date-free approval after later workflow edits."""
    import hashlib
    import json
    from dataclasses import asdict
    from pathlib import Path

    import mlflow

    from skyulf.integrations.databricks import local_approval, local_workflow

    frame = _frame()
    reader = Mock(return_value=frame)
    monkeypatch.setattr(retraining, "read_training_snapshot", reader)
    monkeypatch.setattr(local_workflow, "read_training_snapshot", reader)
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("random", artifact_location=(tmp_path / "runs").as_uri())
    config = {
        "engine": engine,
        "training_table": "workspace.test.labels",
        "training_version": 4,
        "record_key_columns": ["tenant", "id"],
        "input_columns": ["x"],
        "target_column": "target",
        "max_rows": 100,
        "max_input_mb": 1,
        "model_name": "random_model",
        "tracking_uri": uri,
        "registry_uri": uri,
        "score_model_selection": "champion",
        "promotion_policy": "manual_approval",
        "metric": "heldout_rmse",
        "min_improvement": 0.0,
        "quality_threshold": 0.01,
        "pipeline": {
            "preprocessing": [
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}
            ],
            "modeling": {"type": "linear_regression"},
        },
    }
    expected_train, expected_heldout, _ = retraining.split_labeled_snapshot(frame, _spec())
    original_fit = retraining.fit_local_workflow
    observed_fit_rows = []

    def fit_only_training(config, data, **options):
        """Observe the actual preprocessing/model input while retaining the real Core fit."""
        actual = data.train.to_pandas() if isinstance(data.train, pl.DataFrame) else data.train
        pd.testing.assert_frame_equal(actual, expected_train)
        assert set(actual.x).isdisjoint(expected_heldout.x)
        assert len(data.test) == 0
        observed_fit_rows.append(len(actual))
        return original_fit(config, data, **options)

    monkeypatch.setattr(retraining, "fit_local_workflow", fit_only_training)
    candidate = local_workflow.run_action(
        None, config, "train", experiment_name="random", artifact_path=tmp_path / "artifact"
    )
    digest = hashlib.sha256(
        json.dumps(asdict(candidate.comparison), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    assert observed_fit_rows == [16]
    saved_path = client.download_artifacts(
        candidate.run_id, "candidate_training_spec.json", str(tmp_path)
    )
    saved = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    client.log_dict(
        candidate.run_id, {**saved, "holdout_key_sha256": "a" * 64}, "candidate_training_spec.json"
    )
    with pytest.raises(ValueError, match="Saved training snapshot"):
        local_approval._load_evidence(client, "random_model", candidate.model_version, digest)
    client.log_dict(candidate.run_id, saved, "candidate_training_spec.json")
    config.update(
        training_table="workspace.changed.table",
        training_version=999,
        random_state=91,
        test_size=0.5,
        input_columns=["changed"],
        target_column="changed_target",
        engine="polars" if engine == "pandas" else "pandas",
    )
    args = {
        "candidate_version": candidate.model_version,
        "comparison_sha256": digest,
        "expected_champion_version": None,
    }
    changed = frame.copy()
    changed["id"] += 100
    reader.return_value = changed
    with pytest.raises(ValueError, match="membership"):
        local_approval.approve_local_candidate(None, config, **args)
    reader.return_value = frame.iloc[::-1]
    receipt = local_approval.approve_local_candidate(None, config, **args)
    assert receipt.new_version == candidate.model_version
    pinned = reader.call_args.args[1]
    assert pinned.table == "workspace.test.labels" and pinned.version == 4
    assert pinned.random_state == 42 and pinned.test_size == 0.2
    assert pinned.holdout_key_sha256 == candidate.holdout_key_sha256
    assert local_approval.approve_local_candidate(None, config, **args) == receipt
    assert len(client.search_model_versions("name = 'random_model'")) == 1
