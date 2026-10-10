"""Monitoring reconstructs real registered training references without fitting again."""

import pytest
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - shared real-store fixture


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_registered_reference_replays_training_membership(staged, monkeypatch, engine):
    """Persisted training evidence must validate for both supported local engines."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_reference
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )

    _, _, config, _, frame = staged
    config["engine"] = engine
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    _call(staged, "train_register", prepared.reference)
    monitor = MonitorConfig(
        environment="test",
        project="reference",
        model_name=config["model_name"],
        model_version="1",
        source_table=config["training_table"],
        prediction_table="workspace.test.predictions",
    )
    monkeypatch.setattr(
        monitoring_reference, "read_training_snapshot", lambda spark, spec: frame.copy()
    )
    artifact, spec, reference, evidence = monitoring_reference.load_monitoring_reference(
        None, monitor, tracking_uri=config["tracking_uri"], registry_uri=config["registry_uri"]
    )
    assert artifact.manifest.fitted_engine == engine
    assert 0 < len(reference) < len(frame)
    assert spec.version == 4
    assert evidence["model_version"] == "1"
    frame["id"] += 1000
    with pytest.raises(ValueError):
        monitoring_reference.load_monitoring_reference(
            None, monitor, tracking_uri=config["tracking_uri"], registry_uri=config["registry_uri"]
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_holdout_performance_baseline_uses_saved_model_and_verified_membership(
    staged, monkeypatch, engine
):
    """The baseline must be recomputed with identical scoring conventions on heldout rows."""
    import numpy as np
    from test_monitoring_performance import policy

    from skyulf.inference.pipeline_scoring import score_pipeline
    from skyulf.integrations.databricks.observability.monitoring import monitoring_reference
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )
    from skyulf.integrations.databricks.training.fitting.candidate import split_labeled_snapshot

    _, _, config, _, frame = staged
    config["engine"] = engine
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    _call(staged, "train_register", prepared.reference)
    monitor = MonitorConfig(
        environment="test",
        project="reference",
        model_name=config["model_name"],
        model_version="1",
        source_table=config["training_table"],
        prediction_table="workspace.test.predictions",
        label_table="workspace.test.labels",
        result_available_at_column="available",
        performance_policy=policy()
        | {"baseline": {"kind": "training_holdout", "model_version": "1"}},
    )
    monkeypatch.setattr(
        monitoring_reference, "read_training_snapshot", lambda spark, spec: frame.copy()
    )
    artifact, spec, _, evidence = monitoring_reference.load_monitoring_reference(
        None, monitor, tracking_uri=config["tracking_uri"], registry_uri=config["registry_uri"]
    )
    _, heldout, _ = split_labeled_snapshot(frame, spec, engine=engine)
    guesses = score_pipeline(heldout.loc[:, list(artifact.manifest.input_columns)], artifact)
    expected = np.abs(
        guesses["prediction"].to_numpy() - heldout[spec.target_column].to_numpy()
    ).mean()
    assert evidence["performance_baseline"]["value"] == pytest.approx(expected)
    assert evidence["performance_baseline"]["labeled_rows"] == len(heldout)
