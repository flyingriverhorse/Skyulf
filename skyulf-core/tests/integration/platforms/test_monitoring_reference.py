"""Monitoring reconstructs real registered training references without fitting again."""

import pytest
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - shared real-store fixture


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_registered_reference_replays_training_membership(staged, monkeypatch, engine):
    """Persisted training evidence must validate for both supported local engines."""
    from skyulf.integrations.databricks import monitoring_reference
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

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
