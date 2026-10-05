"""Bundle split policies preserve metadata and isolate deployable holdout evidence."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
from skyulf.integrations.databricks.jobs.shared.job_output import _nested_search_output
from skyulf.integrations.databricks.lifecycle._lifecycle_data import load_frame, save_frame
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
from skyulf.integrations.databricks.training.fitting.local_retraining import (
    LocalTrainingSpec,
    _materialize_training_rows,
    split_labeled_snapshot,
)
from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec
from skyulf.integrations.databricks.training.tuning.local_search import prepare_search_pipeline
from skyulf.integrations.databricks.training.tuning.local_search_results import tuning_evidence


def _search():
    """Use a bounded binary search with a training-only decision threshold."""
    return {
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "logistic_regression", "params": {}},
            "strategy": "grid",
            "metric": "f1",
            "search_space": {"C": [1.0]},
            "tune_threshold": True,
        }
    }


def test_nested_time_settings_reach_core_search():
    """The wrapper must preserve chronology settings and threshold opt-in."""
    cv = LocalCVSpec.from_workflow(
        {
            "cv_enabled": True,
            "cv_type": "nested_cv",
            "cv_nested_type": "time_series_split",
            "cv_shuffle": False,
            "cv_gap": 2,
            "cv_test_size": 4,
            "cv_max_train_size": 30,
        }
    )
    model = prepare_search_pipeline(_search(), cv, target_column="target", event_column="event")[
        "modeling"
    ]
    assert model["cv_nested_type"] == "time_series_split"
    assert model["cv_time_column"] == "event"
    assert (model["cv_gap"], model["cv_test_size"], model["cv_max_train_size"]) == (2, 4, 30)
    assert model["tune_threshold"] is True


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_holdout_retains_only_training_split_metadata(engine):
    """Repeated entities cannot leak into final holdout or model feature columns."""
    frame = pd.DataFrame(
        {
            "id": range(60),
            "entity": [n // 3 for n in range(60)],
            "x": range(60),
            "target": [n % 2 for n in range(60)],
        }
    )
    frame.loc[3, "x"] = float("nan")
    spec = LocalTrainingSpec(
        table="workspace.test.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=100,
        max_bytes=100000,
        group_column="entity",
        pre_split_steps=(
            {"name": "eligible", "transformer": "DropMissingRows", "params": {"subset": ["x"]}},
        ),
    )
    train, holdout, _ = split_labeled_snapshot(frame, spec, engine=engine)
    heldout_groups = set(frame.set_index("x").loc[holdout.x, "entity"])
    assert set(train.entity).isdisjoint(heldout_groups)
    assert list(holdout.columns) == ["x", "target"]
    assert "entity" in spec.source_columns
    assert len(train) + len(holdout) == 59
    assert holdout.attrs["group_split"]["training_groups"] == train.entity.nunique()


def test_group_metadata_is_required_before_fitting():
    """A group policy without an identity column must fail before source access."""
    with pytest.raises(ValueError, match="cv_group_column"):
        LocalCVSpec(enabled=True, method="nested_cv", nested_type="group_k_fold")


def test_source_materialization_preserves_group_identifiers():
    """Group identities are categorical values, even when source dates need conversion."""
    spec = LocalTrainingSpec(
        table="workspace.test.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=10,
        max_bytes=100000,
        group_column="entity",
    )
    selected = Mock()
    selected.toLocalIterator.return_value = iter(
        [{"id": 1, "x": 2.0, "target": 4.0, "entity": "customer-a"}]
    )
    result = _materialize_training_rows(selected, spec, spec.source_columns)
    assert result.entity.to_list() == ["customer-a"]


def test_staged_parquet_retains_aligned_split_metadata(tmp_path, monkeypatch):
    """Separate lifecycle tasks must receive the exact filtered metadata and its receipts."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frame = pd.DataFrame(
        {
            "x": [3, 7],
            "entity": ["a", "b"],
            "event": pd.date_range("2026-01-01", periods=2, tz="UTC"),
        }
    )
    frame.attrs["group_split"] = {"training_groups": 2, "holdout_groups": 1}
    store = Mock()
    payload = {}

    def log_artifact(run, path, directory):
        """Keep the exact stage bytes after the writer's temporary file disappears."""
        payload["data"] = Path(path).read_bytes()

    def download_artifacts(run, path, directory):
        """Return the persisted stage to the next task's isolated temporary directory."""
        target = Path(directory) / "train.parquet"
        target.write_bytes(payload["data"])
        return str(target)

    store.client.log_artifact.side_effect = log_artifact
    store.client.download_artifacts.side_effect = download_artifacts
    receipt = save_frame(store, "train", frame)
    restored = load_frame(store, "train", receipt)
    pd.testing.assert_frame_equal(restored, frame)
    assert restored.attrs == frame.attrs


def test_nested_temporal_rejects_random_final_holdout(workflow_config):
    """Offline preview must reject future leakage before reading a source snapshot."""
    workflow_config.update(
        cv_enabled=True,
        cv_type="nested_cv",
        cv_nested_type="time_series_split",
        cv_shuffle=False,
        split_strategy="random",
        holdout_start=None,
        test_size=0.2,
        random_state=42,
        stratify=False,
    )
    with pytest.raises(ValueError, match="temporal final holdout"):
        validate_workflow_config(workflow_config, action="train")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["group_k_fold", "time_series_split"])
def test_nested_policy_artifact_retains_features_only(tmp_path, engine, policy):
    """Split metadata must survive nested search but never become inference inputs."""
    frame = pd.DataFrame(
        {
            "x": range(96),
            "target": [n * 2 for n in range(96)],
            "entity": [n // 4 for n in range(96)],
            "event": pd.date_range("2026-01-01", periods=96, tz="UTC"),
        }
    )
    temporal = policy == "time_series_split"
    frame = frame.drop(columns=["entity" if temporal else "event"])
    cv = LocalCVSpec(
        enabled=True,
        folds=2,
        inner_folds=2,
        method="nested_cv",
        nested_type=policy,
        shuffle=False,
        group_column=None if temporal else "entity",
        gap=1 if temporal else 0,
    )
    pipeline = {
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression", "params": {}},
            "strategy": "grid",
            "metric": "rmse",
            "search_space": {"alpha": [0.1, 1.0]},
        }
    }
    config = prepare_search_pipeline(
        pipeline, cv, target_column="target", event_column="event" if temporal else None
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100000,
    )
    assert artifact.manifest.input_columns == ("x",)
    evidence = tuning_evidence(artifact)
    assert evidence is not None and len(evidence["nested_cv"]["folds"]) == 2
    assert evidence["modeling"]["cv_nested_type"] == policy
    loaded = load_local_pipeline(tmp_path / "artifact")
    pd.testing.assert_frame_equal(
        predict_local_pipeline(frame[["x"]], artifact), predict_local_pipeline(frame[["x"]], loaded)
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_nested_binary_threshold_is_saved_and_enabled(tmp_path, engine):
    """The deployed artifact must apply its independently selected training threshold."""
    frame = pd.DataFrame(
        {"x": range(72), "target": ["yes" if n % 4 == 0 else "no" for n in range(72)]}
    )
    cv = LocalCVSpec(enabled=True, folds=3, inner_folds=2, method="nested_cv")
    config = prepare_search_pipeline(_search(), cv, target_column="target", event_column=None)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100000,
    )
    evidence = tuning_evidence(artifact)
    assert artifact.manifest.use_tuned_thresholds
    assert evidence is not None and evidence["decision_thresholds"] is not None
    assert all(
        fold["threshold_selection"]["decision_thresholds"]
        for fold in evidence["nested_cv"]["folds"]
    )
    readable = "".join(_nested_search_output(evidence))
    assert "Final decision thresholds" in readable and "Decision thresholds" in readable
    assert "f1" in readable
    loaded = load_local_pipeline(tmp_path / "artifact")
    pd.testing.assert_frame_equal(
        predict_local_pipeline(frame[["x"]], artifact), predict_local_pipeline(frame[["x"]], loaded)
    )
