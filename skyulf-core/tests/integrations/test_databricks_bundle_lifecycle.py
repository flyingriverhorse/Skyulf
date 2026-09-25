"""Exercise generated lifecycle orchestration against a real local MLflow store."""

import pandas as pd
import pytest

from skyulf.integrations.databricks import job_runtime, local_retraining, local_workflow

mlflow = pytest.importorskip("mlflow")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("selection", ["pinned_version", "champion"])
def test_bundle_nomination_comparison_and_promotion_are_separate(
    tmp_path, monkeypatch, engine, selection
):
    """Real artifacts must retain a tied contender and survive failed comparison on either engine."""
    frame = pd.DataFrame(
        {
            "id": range(12),
            "x": [float(value) for value in range(12)],
            "target": [10.0 + 2 * value for value in range(12)],
            "event_time": pd.to_datetime(["2026-01-10"] * 8 + ["2026-02-10"] * 4, utc=True),
            "label_at": pd.to_datetime(["2026-01-11"] * 8 + ["2026-02-11"] * 4, utc=True),
        }
    )
    monkeypatch.setattr(local_retraining, "read_training_snapshot", lambda *args: frame)
    monkeypatch.setattr(local_workflow, "read_training_snapshot", lambda *args: frame)
    store = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=store, registry_uri=store)
    client.create_experiment("lifecycle", artifact_location=(tmp_path / "mlruns").as_uri())
    config = {
        "engine": engine,
        "training_table": "workspace.test.source",
        "training_version": 0,
        "model_name": "lifecycle_model",
        "tracking_uri": store,
        "registry_uri": store,
        "score_model_selection": selection,
        "promotion_policy": "automatic",
        "score_handoff": "after_alias_change",
        "model_version": "1",
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "target",
        "event_column": "event_time",
        "result_available_at_column": "label_at",
        "start": "2026-01-01T00:00:00+00:00",
        "holdout_start": "2026-02-01T00:00:00+00:00",
        "cutoff": "2026-03-01T00:00:00+00:00",
        "max_rows": 20,
        "max_bytes": 100_000,
        "metric": "heldout_rmse",
        "min_improvement": 0.1,
        "quality_threshold": 100.0,
        "pipeline": {
            "preprocessing": [],
            "modeling": {
                "type": "linear_regression",
                "params": {"fit_intercept": False},
            },
        },
    }

    def run(version):
        """Use the exact template action with only its Spark input boundary replaced."""
        outcome = job_runtime.run_bundle_action(
            None,
            config,
            {"lifecycle_action": "train"},
            task_role="lifecycle",
            experiment_name="lifecycle",
            artifact_path=tmp_path / f"model-{version}",
        )
        assert outcome.score_requested == (version in {1, 2})
        return outcome.result

    first = run(1)
    assert first.alias_change.kind == "initial"
    config["pipeline"]["modeling"]["params"]["fit_intercept"] = True
    second = run(2)
    assert second.alias_change.kind == "promotion"
    third = run(3)
    assert third.alias_change is None
    assert str(client.get_model_version_by_alias("lifecycle_model", "champion").version) == "2"
    assert str(client.get_model_version_by_alias("lifecycle_model", "challenger").version) == "3"
    assert client.get_model_version("lifecycle_model", "3").tags["validation_status"] == "rejected"
    config["promotion_policy"] = "manual_approval"
    fourth = run(4)
    assert fourth.model_version == "4" and config["model_version"] == "1"
    assert str(client.get_model_version_by_alias("lifecycle_model", "champion").version) == "2"
    assert str(client.get_model_version_by_alias("lifecycle_model", "challenger").version) == "4"

    def comparison_fails(*args, **kwargs):
        """Fail after nomination to preserve evidence for a registered contender."""
        raise RuntimeError("comparison unavailable")

    monkeypatch.setattr(local_retraining, "compare_registered_local_models", comparison_fails)
    with pytest.raises(RuntimeError, match="comparison unavailable"):
        run(5)
    assert str(client.get_model_version_by_alias("lifecycle_model", "champion").version) == "2"
    assert str(client.get_model_version_by_alias("lifecycle_model", "challenger").version) == "5"
    assert client.get_model_version("lifecycle_model", "5").tags["validation_status"] == "error"
