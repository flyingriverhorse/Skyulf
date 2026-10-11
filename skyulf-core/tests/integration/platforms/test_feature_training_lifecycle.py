"""Keep native lookup settings attached to every bounded training invocation."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.lifecycle.workflow import (
    resolve_target_config,
    resolve_training_spec,
    training_spec,
)
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec


def _config(workflow_config):
    """Require historical lookups with a source timestamp distinct from split metadata."""
    workflow_config.update(
        engine="pandas",
        inference_mode="spark",
        feature_lookup={
            "lookups": [
                {
                    "table_name": "workspace.features.history",
                    "lookup_key": ["company_id"],
                    "feature_names": ["x"],
                    "timestamp_lookup_key": "observed_at",
                }
            ]
        },
    )
    return workflow_config


def test_training_spec_retains_lookup_controls_and_old_dataset_identity(workflow_config):
    """Lookups cannot disappear before data loading; absent lookups keep old identities."""
    old = training_spec(workflow_config)
    spec = training_spec(_config(workflow_config))
    assert spec.feature_lookup_json is not None
    assert json.loads(spec.feature_lookup_json)["lookups"][0]["feature_names"] == ["x"]
    assert {"company_id", "observed_at"}.issubset(spec.source_columns)
    assert spec.dataset_id != old.dataset_id
    assert (
        replace(old, feature_lookup_json=None, feature_binding_json=None).dataset_id
        == old.dataset_id
    )


@pytest.mark.parametrize("field,value", [("engine", "polars"), ("inference_mode", "local")])
def test_lookup_requires_admitted_training_and_spark_scoring(workflow_config, field, value):
    """Unsupported modes must fail before any remote feature or model operation."""
    config = _config(workflow_config)
    config[field] = value
    with pytest.raises(ValueError, match="feature_lookup"):
        training_spec(config)


def test_lookup_tables_use_target_bindings(workflow_config):
    """Feature tables must resolve with the selected project target rather than stale literals."""
    config = _config(workflow_config)
    config["feature_lookup"]["lookups"][0]["table_name"] = "{catalog}.{input_schema}.history"
    bindings = {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
    }
    resolved = resolve_target_config(config, bindings)
    assert resolved["feature_lookup"]["lookups"][0]["table_name"] == "workspace.test.history"
    assert (
        config["feature_lookup"]["lookups"][0]["table_name"] == "{catalog}.{input_schema}.history"
    )


def test_training_initialization_pins_feature_tables(workflow_config, monkeypatch):
    """Task separation needs the same frozen table bindings from prepare through registration."""
    from skyulf.integrations.databricks.feature_store import training

    config = _config(workflow_config)
    expected = training_spec(config)
    pin = Mock(return_value=expected)
    monkeypatch.setattr(training, "pin_training_lookup", pin)
    resolved = resolve_training_spec(Mock(), config, datetime(2026, 3, 1, tzinfo=UTC))
    pin.assert_called_once()
    assert resolved == expected


def test_tampered_training_spec_lookup_is_rejected(workflow_config):
    """Persisted contracts must validate even when callers bypass workflow parsing."""
    spec = training_spec(_config(workflow_config))
    with pytest.raises(ValueError):
        replace(spec, feature_lookup_json='{"lookups": []}')
    assert isinstance(spec, TrainingSpec)


def _online_config(workflow_config):
    """Declare latest feature training without a batch scoring requirement."""
    config = _config(workflow_config)
    config["inference_mode"] = "local"
    config["input_columns"] = ["x", "fresh_at"]
    lookup = config["feature_lookup"]["lookups"][0]
    lookup.pop("timestamp_lookup_key")
    lookup["feature_names"] = ["x", "fresh_at"]
    config["online_features"] = {
        "required_features": ["x", "fresh_at"],
        "freshness": [{"column": "fresh_at", "max_age_seconds": 3600.0}],
    }
    return config


def test_online_policy_survives_training_spec_roundtrip_without_batch_sink(workflow_config):
    """Preparation and fitting must retain the guard even when no batch job runs."""
    from dataclasses import asdict

    config = _online_config(workflow_config)
    config.pop("prediction_table")
    spec = training_spec(config)
    assert spec.online_policy_json is not None
    policy = json.loads(spec.online_policy_json)
    assert policy["freshness"] == [["fresh_at", 3600.0]]
    payload = asdict(spec)
    for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
        value = payload[field]
        payload[field] = value.isoformat() if value is not None else None
    assert TrainingSpec.from_payload(payload).online_policy_json == spec.online_policy_json


@pytest.mark.parametrize("invalid", ["polars", "temporal", "model_set", "missing_lookup"])
def test_online_training_rejects_unsupported_combinations(workflow_config, invalid):
    """Online configuration cannot silently weaken the existing endpoint boundary."""
    config = _online_config(workflow_config)
    if invalid == "polars":
        config["engine"] = "polars"
    elif invalid == "temporal":
        config["feature_lookup"]["lookups"][0]["timestamp_lookup_key"] = "observed_at"
    elif invalid == "model_set":
        config["training_layout"] = "multi_target"
    else:
        config.pop("feature_lookup")
    with pytest.raises(ValueError, match="online|Online"):
        training_spec(config)


def test_training_logger_transports_saved_online_policy(workflow_config, monkeypatch):
    """The final logger must receive the exact policy carried by preparation and fit."""
    pytest.importorskip("mlflow")

    from skyulf.integrations.databricks.feature_store import snapshots, training
    from skyulf.integrations.databricks.feature_store.lifecycle_config import (
        binding_json,
        serialize_feature_spec,
    )
    from skyulf.integrations.mlflow.models import feature_model

    spec = training_spec(_online_config(workflow_config))
    lookup = training.training_lookup(spec)
    assert lookup is not None
    binding = {
        "version": 1,
        "lookup_spec": serialize_feature_spec(lookup),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {"table_name": lookup.lookups[0].table_name, "table_id": "delta-id", "version": 1}
            ],
        },
    }
    spec = replace(spec, feature_binding_json=binding_json(binding))
    native = object()
    monkeypatch.setattr(training, "logging_training_set", lambda *args: native)
    monkeypatch.setattr(snapshots, "validate_snapshots", lambda *args: None)
    logger = Mock(return_value="runs:/run/model")
    monkeypatch.setattr(feature_model, "log_feature_pipeline_model", logger)
    assert (
        training.log_training_feature_model(
            "fitted", spark=object(), spec=spec, run_id="run", tracking_uri="databricks"
        )
        == "runs:/run/model"
    )
    assert spec.online_policy_json is not None
    assert logger.call_args.kwargs["online_policy"].to_dict() == json.loads(spec.online_policy_json)
    assert logger.call_args.kwargs["training_set"] is native


def test_model_set_packaging_cannot_drop_component_online_policy(workflow_config):
    """Multi-target packaging must reject untransported component policy before registry reads."""
    from skyulf.integrations.databricks.model_sets.model_set_project import (
        package_training_model_set,
    )
    from skyulf.integrations.databricks.training.branches import (
        BranchTrainingResult,
        TrainingBranch,
    )

    spec = training_spec(_online_config(workflow_config))
    branch = TrainingBranch(
        name="main", spec=spec, pipeline={}, model_name="main.models.online", metric="rmse"
    )
    outcome = BranchTrainingResult("run", spec.table, spec.version, {}, "a" * 64)
    with pytest.raises(ValueError, match="online|Online"):
        package_training_model_set(
            None,
            (branch,),
            outcome,
            {},
            composition_source="",
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"required_features": []},
        {
            "required_features": ["x", "fresh_at"],
            "freshness": [{"column": "fresh_at", "max_age_seconds": None}],
        },
    ],
)
def test_incomplete_online_policy_fails_before_training(workflow_config, policy):
    """Editable YAML placeholders must never become an unguarded registered model."""
    config = _online_config(workflow_config)
    config["online_features"] = policy
    with pytest.raises(ValueError):
        training_spec(config)
