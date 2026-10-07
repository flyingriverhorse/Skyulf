"""Keep native lookup settings attached to every bounded training invocation."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.lifecycle.local_workflow import (
    resolve_target_config,
    resolve_training_spec,
    training_spec,
)
from skyulf.integrations.databricks.training.fitting.local_retraining import LocalTrainingSpec


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
    assert isinstance(spec, LocalTrainingSpec)
