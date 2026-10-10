"""Model sets preserve compatible native feature lookup lineage across components."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.integration.platforms.test_feature_native_scoring import _binding


def _component(features=("amount",), *, table="catalog.schema.features", version=7):
    """Describe one independently trained component on a pinned feature table."""
    binding = _binding()
    binding["lookup_spec"]["lookups"][0]["feature_names"] = list(features)
    binding["lookup_spec"]["lookups"][0]["table_name"] = table
    binding["lookup_evidence"]["feature_tables"][0].update(table_name=table, version=version)
    return SimpleNamespace(
        feature_lookup_json=json.dumps(binding),
        manifest=SimpleNamespace(input_columns=features),
    )


def test_union_keeps_compatible_features_once_and_preserves_snapshot():
    """Shared fetched columns may be deduplicated only when their complete lineage agrees."""
    from skyulf.integrations.databricks.feature_store.model_sets import union_feature_binding

    first, second = _component(("amount", "risk")), _component(("risk", "score"))
    binding = union_feature_binding(
        {"first": first, "second": second}, ("id", "amount", "risk", "score")
    )
    assert binding is not None
    assert binding["lookup_spec"]["label"] is None
    assert binding["lookup_spec"]["exclude_columns"] == ["entity", "event_time"]
    assert [
        name for lookup in binding["lookup_spec"]["lookups"] for name in lookup["feature_names"]
    ] == ["amount", "risk", "score"]
    assert binding["lookup_evidence"] == _binding()["lookup_evidence"]


@pytest.mark.parametrize("change", ["table", "snapshot", "timestamp", "direct"])
def test_union_rejects_incompatible_component_lineage(change):
    """A complete set cannot silently replace one branch's trained feature meaning."""
    from skyulf.integrations.databricks.feature_store.model_sets import union_feature_binding

    first, second = _component(), _component()
    changed = deepcopy(json.loads(second.feature_lookup_json))
    if change == "table":
        changed["lookup_spec"]["lookups"][0]["table_name"] = "catalog.schema.other"
        changed["lookup_evidence"]["feature_tables"][0]["table_name"] = "catalog.schema.other"
    elif change == "snapshot":
        changed["lookup_evidence"]["feature_tables"][0]["version"] = 8
    elif change == "timestamp":
        changed["lookup_spec"]["lookups"][0]["timestamp_lookup_key"] = "different_time"
    else:
        second.feature_lookup_json = None
    if change != "direct":
        second.feature_lookup_json = json.dumps(changed)
    with pytest.raises(ValueError, match="incompatible|snapshot|direct"):
        union_feature_binding({"first": first, "second": second}, ("id", "amount"))


def test_ordinary_model_set_never_acquires_feature_lookup():
    """Existing component sets retain the raw package path and metadata contract."""
    from skyulf.integrations.databricks.feature_store.model_sets import union_feature_binding

    component = _component()
    component.feature_lookup_json = None
    assert union_feature_binding({"ordinary": component}, ("id", "amount")) is None


def test_component_binding_must_match_completed_training_spec():
    """Set packaging must not replace the saved branch plan with registry lookup metadata."""
    from skyulf.integrations.databricks.feature_store.model_sets import validate_component_bindings

    component = _component()
    spec = SimpleNamespace(
        feature_binding_json=component.feature_lookup_json, input_columns=("amount",)
    )
    branch = SimpleNamespace(name="first", spec=spec)
    validate_component_bindings((branch,), {"first": component})
    spec.feature_binding_json = _component(version=8).feature_lookup_json
    with pytest.raises(ValueError, match="training plan"):
        validate_component_bindings((branch,), {"first": component})


def test_set_logger_uses_native_union_instead_of_discarding_binding(tmp_path, monkeypatch):
    """One native model-set envelope must contain the complete compatible lookup union."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.feature_store import snapshots
    from skyulf.integrations.databricks.model_sets import model_set_project as project
    from skyulf.integrations.mlflow.models import feature_model

    source, artifact, training_set = object(), object(), object()
    binding = _binding()
    lookup = Mock(return_value=training_set)
    log = Mock(return_value="runs:/run/set")
    monkeypatch.setattr(project, "model_set_training_set", lookup, raising=False)
    monkeypatch.setattr(feature_model, "log_feature_model_set", log)
    monkeypatch.setattr(snapshots, "validate_snapshots", Mock())
    result = project._log_training_set_package(
        None, source, artifact, tmp_path, binding, "run", "set", "tracking"
    )
    lookup.assert_called_once_with(None, source, artifact, binding)
    assert result == "runs:/run/set"
    assert log.call_args.kwargs["training_set"] is training_set
    assert log.call_args.kwargs["lookup_binding"] == binding


def test_set_logger_rechecks_feature_version_before_registration(tmp_path, monkeypatch):
    """Feature drift during packaging cannot produce an admitted set registry candidate."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.feature_store import snapshots
    from skyulf.integrations.databricks.model_sets import model_set_project as project
    from skyulf.integrations.mlflow.models import feature_model

    log = Mock(return_value="runs:/run/set")
    monkeypatch.setattr(project, "model_set_training_set", Mock())
    monkeypatch.setattr(feature_model, "log_feature_model_set", log)
    monkeypatch.setattr(
        snapshots, "validate_snapshots", Mock(side_effect=ValueError("feature drift"))
    )
    with pytest.raises(ValueError, match="feature drift"):
        project._log_training_set_package(
            None, object(), object(), tmp_path, _binding(), "run", "set", "tracking"
        )
    log.assert_called_once()


def test_declared_lookup_cannot_drop_its_training_snapshot():
    """A missing binding cannot silently turn a declared native branch into an ordinary model."""
    from skyulf.integrations.databricks.feature_store.model_sets import validate_component_bindings

    artifact = _component()
    artifact.feature_lookup_json = None
    spec = SimpleNamespace(feature_lookup_json="{}", feature_binding_json=None)
    with pytest.raises(ValueError, match="snapshot"):
        validate_component_bindings(
            (SimpleNamespace(name="first", spec=spec),), {"first": artifact}
        )


def test_approval_enrichment_retains_lookup_controls_and_record_keys(monkeypatch):
    """The functional probe must join saved features before local scoring without excluding IDs."""
    from skyulf.integrations.databricks.feature_store import model_sets

    binding = _binding()
    binding["lookup_spec"].update(label=None, exclude_columns=["entity", "event_time"])
    artifact = SimpleNamespace(
        feature_lookup_json=json.dumps(binding),
        manifest=SimpleNamespace(
            input_schema=[
                SimpleNamespace(name="id", dtype="int64"),
                SimpleNamespace(name="amount", dtype="float64"),
            ]
        ),
    )
    source = SimpleNamespace(
        columns=["id", "entity", "event_time"],
        dtypes=[("event_time", "timestamp")],
        isStreaming=False,
        select=Mock(return_value="selected"),
    )
    training_set = SimpleNamespace(load_df=Mock(return_value="enriched"))
    create = Mock(return_value=training_set)
    monkeypatch.setattr(model_sets, "create_checked_training_set", create)
    monkeypatch.setattr(model_sets, "validate_feature_snapshot", lambda *args: {})
    result = model_sets.enrich_approval_source(None, source, artifact)
    source.select.assert_called_once_with("id", "entity", "event_time")
    assert create.call_args.args[1].exclude_columns == ()
    assert result == "enriched"
