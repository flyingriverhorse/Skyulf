"""Internal contract failures must reject before logging or incremental publication."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.feature_store import lifecycle_config, training
from skyulf.integrations.databricks.features import delta, runtime
from skyulf.integrations.databricks.features.config import FeatureGroup, FeaturePlan
from skyulf.integrations.databricks.features.graph import plan_digest
from skyulf.integrations.databricks.projects.yaml_config import read_yaml_mapping
from skyulf.integrations.databricks.scoring.incremental import incremental_batch


def test_workflow_lookup_requires_a_parsed_spec(monkeypatch):
    """An enabled lookup must not serialize a missing parser result as valid configuration."""
    monkeypatch.setattr(lifecycle_config, "parse_lookup_config", lambda *args, **kwargs: None)
    serializer = Mock(return_value={})
    monkeypatch.setattr(lifecycle_config, "serialize_feature_spec", serializer)
    with pytest.raises(ValueError, match="enabled feature lookup"):
        lifecycle_config.workflow_lookup_json(
            {
                "feature_lookup": {},
                "engine": "pandas",
                "inference_mode": "spark",
                "target_column": "target",
                "input_columns": ["x"],
            }
        )
    serializer.assert_not_called()


def test_feature_logger_requires_a_training_lookup(monkeypatch):
    """A missing native lookup must not reach the model logger even under optimized Python."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models import feature_model

    monkeypatch.setattr(training, "parse_feature_binding", lambda value: {})
    monkeypatch.setattr(training, "training_lookup", lambda spec: None)
    monkeypatch.setattr(training, "logging_training_set", lambda *args: object())
    logger = Mock()
    monkeypatch.setattr(feature_model, "log_feature_pipeline_model", logger)
    with pytest.raises(ValueError, match="requires a feature lookup"):
        training.log_training_feature_model(
            "fitted",
            spark=object(),
            spec=SimpleNamespace(feature_binding_json="{}"),
            run_id="run",
            tracking_uri="databricks",
        )
    logger.assert_not_called()


def test_empty_increment_requires_a_fitted_pipeline_artifact():
    """A lookalike artifact cannot bypass fitted-state validation for an empty batch."""
    artifact = SimpleNamespace(
        pipeline=SimpleNamespace(feature_engineer=SimpleNamespace(fitted_steps=[]))
    )
    with pytest.raises(TypeError, match="FittedPipelineArtifact"):
        incremental_batch._empty_incremental_predictions(Mock(artifact=artifact), ("prediction",))


@pytest.mark.parametrize("name", ["features; DROP TABLE data", "features`--", "a.b.c.d"])
def test_feature_publication_rejects_sql_identifiers_before_access(name):
    """Untrusted table names cannot reach a feature read, temporary view or SQL statement."""
    spark = Mock()
    frame = Mock()
    with pytest.raises(ValueError, match="simple identifiers"):
        delta.publish_feature_table(spark, frame, name, ("id",), "at")
    assert spark.mock_calls == [] and frame.mock_calls == []


def test_yaml_safe_loader_rejects_python_object_construction(tmp_path):
    """The duplicate-checking SafeLoader must never instantiate Python objects from YAML tags."""
    pytest.importorskip("yaml")
    marker = tmp_path / "executed"
    path = tmp_path / "unsafe.yml"
    path.write_text(
        "value: !!python/object/apply:pathlib.Path.touch\n"
        f"  - !!python/object/apply:pathlib.Path ['{marker.as_posix()}']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid YAML"):
        read_yaml_mapping(path)
    assert not marker.exists()


def test_feature_transform_digest_is_checked_before_execution(tmp_path):
    """Changed trusted project source must not run under an earlier job's pinned identity."""
    group = FeatureGroup(
        "company",
        "catalog.schema.raw",
        "catalog.schema.features",
        "src/features/groups/company.py:compute",
        ("size",),
    )
    plan = FeaturePlan("catalog.schema.base", "catalog.schema.merged", ("id",), "at", (group,))
    path = tmp_path / group.transform.split(":")[0]
    path.parent.mkdir(parents=True)
    path.write_text("def compute(frame):\n    return frame\n", encoding="utf-8")
    snapshot = {
        "plan_sha256": plan_digest(plan),
        "selected": ["company"],
        "transform_sha256": {"company": runtime.transform_source(tmp_path, group)[1]},
    }
    path.write_text("raise AssertionError('source executed')\n", encoding="utf-8")
    spark = Mock()
    with pytest.raises(ValueError, match="transform changed"):
        runtime.build_feature_group(spark, tmp_path, plan, snapshot, "company")
    assert spark.mock_calls == []
