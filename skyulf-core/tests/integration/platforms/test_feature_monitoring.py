"""Monitor the same native feature lineage that produced saved predictions."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.feature_store.config import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
)
from skyulf.integrations.databricks.feature_store.lifecycle_config import serialize_feature_spec
from skyulf.integrations.databricks.feature_store.monitoring import (
    bind_observation_reader,
    enrich_observation,
)


def _binding():
    """Retain one historical feature declaration from the trained component."""
    spec = FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="workspace.features.history",
                lookup_key=("entity",),
                feature_names=("x",),
            ),
        ),
        label="target",
        exclude_columns=("id", "entity"),
    )
    return {
        "version": 1,
        "lookup_spec": serialize_feature_spec(spec),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {"table_name": "workspace.features.history", "table_id": "fixed", "version": 4}
            ],
        },
    }


def test_monitoring_fetches_features_for_scored_keys_without_requiring_label(monkeypatch):
    """Unlabeled production rows must still support drift and delayed-label observations."""
    from skyulf.integrations.databricks.feature_store import monitoring

    binding = _binding()
    source = Mock(columns=["id", "entity"])
    native = Mock()
    helper = Mock(return_value=native)
    monkeypatch.setattr(monitoring, "create_checked_training_set", helper)
    result = enrich_observation(
        Mock(), source, ("id",), ("x",), binding, {"feature_lookup": binding}
    )
    assert helper.call_args.args[1].label is None
    assert helper.call_args.args[1].exclude_columns == ()
    assert result is native.load_df.return_value
    source.select.assert_called_once_with("id", "entity")


@pytest.mark.parametrize("change", ["missing", "version", "lookup"])
def test_monitoring_rejects_prediction_lineage_mismatch(change):
    """The drift population cannot be reconstructed using a different feature history."""
    expected = _binding()
    receipt = deepcopy(expected)
    if change == "version":
        receipt["lookup_evidence"]["feature_tables"][0]["version"] += 1
    elif change == "lookup":
        receipt["lookup_spec"]["lookups"][0]["lookup_key"] = ["other"]
    source = Mock()
    with pytest.raises(ValueError, match="feature"):
        enrich_observation(
            Mock(),
            source,
            ("id",),
            ("x",),
            expected,
            {} if change == "missing" else {"feature_lookup": receipt},
        )
    source.select.assert_not_called()


def test_monitoring_reader_requires_distributed_batch_for_native_features():
    """Unsupported local or online observation must fail explicitly before source reads."""
    artifact = SimpleNamespace(feature_lookup_json=json.dumps(_binding()))
    reader = Mock(return_value="result")
    config = SimpleNamespace(execution_engine="spark", serving_endpoint=None)
    bound = bind_observation_reader(reader, config, artifact)
    assert bound() == "result"
    assert reader.call_args.kwargs["feature_binding"] == _binding()
    config.execution_engine = "local"
    with pytest.raises(ValueError, match="Spark batch"):
        bind_observation_reader(reader, config, artifact)


def test_performance_population_tracks_lookup_identity_but_allows_new_versions():
    """Production baselines cannot silently compare a replacement feature table's semantics."""
    from skyulf.integrations.databricks.observability.monitoring.local.monitoring_performance import (
        population_contract,
    )

    evidence = {
        "source_table_id": "s",
        "prediction_table_id": "p",
        "label_table_id": "l",
        "feature_lookup": _binding(),
    }
    original = population_contract("metric", evidence)
    table = evidence["feature_lookup"]["lookup_evidence"]["feature_tables"][0]
    table["version"] += 1
    assert population_contract("metric", evidence) == original
    table["table_id"] = "replacement"
    assert population_contract("metric", evidence) != original
