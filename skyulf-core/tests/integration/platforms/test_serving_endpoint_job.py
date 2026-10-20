"""Owned serving orchestration validates before mutations and proves actual predictions."""

import importlib.util
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("mlflow")

from skyulf.integrations.databricks.jobs.serving_endpoint_job import (  # noqa: E402
    sample_serving_rows as real_sample_serving_rows,
)


def test_serving_endpoint_job_api_is_available():
    """The Bundle notebook needs a real installed entry point rather than template-only code."""
    assert importlib.util.find_spec("skyulf.integrations.databricks.jobs.serving_endpoint_job")


def test_serving_handoff_is_available():
    """Both training layouts and daily promotion must share one receipt-driven policy."""
    assert importlib.util.find_spec("skyulf.integrations.databricks.jobs.serving_handoff")


@pytest.fixture
def handoff():
    """Exercise the actual receipt policy while keeping cloud services out of unit decisions."""
    return import_module("skyulf.integrations.databricks.jobs.serving_handoff")


@pytest.mark.parametrize("kind,prior", [("initial", None), ("promotion", "1"), ("rollback", "3")])
def test_handoff_uses_concrete_changed_champion(handoff, kind, prior):
    """Initial, promoted and restored versions all reach the same serving endpoint."""
    receipt = {
        "kind": kind,
        "alias": "champion",
        "model_name": "main.ml.model",
        "new_version": "2",
        "prior_version": prior,
    }
    result = handoff.serving_transition(receipt, model_name="main.ml.model")
    assert result == "2"


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {},
        {"kind": "rejection"},
        {"kind": "promotion", "alias": "challenger", "new_version": "2"},
        {
            "kind": "promotion",
            "alias": "champion",
            "model_name": "main.ml.model",
            "new_version": "2",
            "prior_version": "2",
        },
    ],
)
def test_handoff_has_no_version_without_a_real_champion_change(handoff, receipt):
    """A pending, rejected or unchanged candidate cannot create a paid endpoint."""
    assert handoff.serving_transition(receipt, model_name="main.ml.model") == ""


def test_handoff_rejects_wrong_model_and_mutable_version(handoff):
    """A valid-looking receipt cannot accidentally serve another model or a moving alias."""
    receipt = {
        "kind": "promotion",
        "alias": "champion",
        "model_name": "other.ml.model",
        "new_version": "2",
        "prior_version": "1",
    }
    with pytest.raises(ValueError, match="model"):
        handoff.serving_transition(receipt, model_name="main.ml.model")
    receipt.update(model_name="main.ml.model", new_version="champion")
    with pytest.raises(ValueError, match="version"):
        handoff.serving_transition(receipt, model_name="main.ml.model")


def test_unselected_handoff_never_reads_optional_config(handoff):
    """Pre-existing generated projects remain compatible without a serving.yml file."""
    values = {}
    saved = {}
    dbutils = SimpleNamespace(
        jobs=SimpleNamespace(
            taskValues=SimpleNamespace(set=lambda key, value: saved.update({key: value}))
        )
    )
    handoff.publish_serving_handoff(dbutils, None, values)
    assert saved == {}


@pytest.fixture
def endpoint_project(tmp_path):
    """Keep generated placeholders intact so tests exercise real target binding."""
    settings = {
        "enabled": True,
        "follow_champion": True,
        "endpoint_name": "sample-endpoint{resource_suffix}",
        "model_name": "{catalog}.{metadata_schema}.sample_model{resource_suffix}",
        "logging_mode": "disabled",
        "logging_catalog": "",
        "logging_schema": "",
        "logging_table_prefix": "",
        "sql_function": {
            "enabled": True,
            "name": "{catalog}.{metadata_schema}.sample_score{resource_suffix}",
        },
        "permissions": [],
    }
    document = {"serving_endpoint": settings}
    path = tmp_path / "serving.yml"
    path.write_text(yaml.safe_dump(document))
    values = {
        "serving_config_path": str(path),
        "catalog": "main",
        "input_schema": "source",
        "output_schema": "scores",
        "metadata_schema": "ml",
        "resource_suffix": "_alice",
        "bundle_project": "sample",
        "bundle_target": "dev",
        "bundle_mode": "development",
        "deployed_serving_endpoint": "true",
        "cost_tag_key": "team",
        "cost_tag_value": "risk",
    }
    return document, path, values


def test_settings_bind_developer_names_without_changing_model_suffix(endpoint_project):
    """Endpoint-safe suffixes and UC-safe suffixes represent the same deployment owner."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    settings = api.read_serving_endpoint_settings(values)
    assert settings["endpoint_name"] == "sample-endpoint-alice"
    assert settings["model_name"] == "main.ml.sample_model_alice"
    assert settings["sql_function"]["name"] == "main.ml.sample_score_alice"
    assert settings["ownership"]["suffix"] == "_alice"
    assert settings["tags"] == {"team": "risk"}


@pytest.mark.parametrize(
    "key,value",
    [("project", "risk"), ("team", ""), ("", "risk"), ("team", "REPLACE_WITH_COST_TAG")],
)
def test_settings_reject_invalid_cost_tags_before_remote_work(endpoint_project, key, value):
    """Attribution must not overwrite ownership tags or silently lose its configured value."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    _, _, values = endpoint_project
    values.update(cost_tag_key=key, cost_tag_value=value)
    with pytest.raises(ValueError, match="tags|tag"):
        api.read_serving_endpoint_settings(values)


@pytest.mark.parametrize("section", ["rollout", "online_feature_lookup"])
def test_settings_refuse_an_endpoint_managed_by_another_optional_job(endpoint_project, section):
    """Single-version serving must not overwrite traffic rollout or online lookup endpoints."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    document[section] = (
        {"monitoring": {"serving_endpoint": "sample-endpoint-alice"}}
        if section == "rollout"
        else {"endpoint": {"endpoint_name": "sample-endpoint-alice"}}
    )
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="already managed"):
        api.read_serving_endpoint_settings(values)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("verification", {"sample_rows": 0}, "verification.sample_rows"),
        ("enabled", "true", "boolean"),
        ("follow_champion", "true", "boolean"),
        ("follow_champion", None, "boolean"),
        ("follow_champion", 1, "boolean"),
        ("selection", "latest", "Unknown serving_endpoint fields.*selection"),
        ("ready_timeout_s", -1, "positive"),
        ("endpoint_name", "x" * 64, "endpoint_name"),
        ("permisions", [], "Unknown"),
    ],
)
def test_invalid_settings_fail_before_any_cloud_access(endpoint_project, field, value, match):
    """Invalid configuration is rejected before model downloads or paid endpoint creation."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    document["serving_endpoint"][field] = value
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=match):
        api.read_serving_endpoint_settings(values)


@pytest.mark.parametrize("field,value", [("selection", "champion"), ("smoke", {"rows": 20})])
def test_settings_reject_removed_serving_options(endpoint_project, field, value):
    """Removed public settings cannot silently survive in newly deployed YAML."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    document["serving_endpoint"][field] = value
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=f"Unknown serving_endpoint fields.*{field}"):
        api.read_serving_endpoint_settings(values)


@pytest.mark.parametrize("sample_rows", [None, 1, 20, 1000])
def test_verification_settings_keep_default_and_accept_sample_bounds(endpoint_project, sample_rows):
    """The renamed control retains the default sample size and both supported bounds."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    if sample_rows is not None:
        document["serving_endpoint"]["verification"] = {"sample_rows": sample_rows}
    path.write_text(yaml.safe_dump(document))
    settings = api.read_serving_endpoint_settings(values)
    assert settings["verification"] == {"sample_rows": 20 if sample_rows is None else sample_rows}
    assert "selection" not in settings and "smoke" not in settings


@pytest.mark.parametrize("sample_rows", [-1, 1001, True, 20.0, "20", None])
def test_verification_rejects_noninteger_or_unbounded_samples(endpoint_project, sample_rows):
    """Verification limits cannot accept coercible scalars or exceed the bounded source read."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    document["serving_endpoint"]["verification"] = {"sample_rows": sample_rows}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(
        ValueError, match="verification.sample_rows must be an integer from 1 to 1000"
    ):
        api.read_serving_endpoint_settings(values)


@pytest.mark.parametrize(
    "verification,match",
    [
        (None, "verification must be a mapping"),
        ([], "verification must be a mapping"),
        ({}, "verification.sample_rows must be an integer"),
        ({"rows": 20}, "Unknown verification fields.*rows"),
        ({"sample_rows": 20, "rows": 20}, "Unknown verification fields.*rows"),
    ],
)
def test_verification_rejects_malformed_blocks_and_removed_nested_rows(
    endpoint_project, verification, match
):
    """Renaming the parent block must not preserve rows as a hidden alias or weaken validation."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    document["serving_endpoint"]["verification"] = verification
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=match):
        api.read_serving_endpoint_settings(values)


@pytest.mark.parametrize("mode", ["development", "production"])
@pytest.mark.parametrize("setting", [False, True])
def test_handoff_follows_yaml_in_every_target_and_keeps_score_independent(
    endpoint_project, handoff, mode, setting
):
    """The YAML switch alone controls dispatch; development follows the champion too."""
    document, path, values = endpoint_project
    document["serving_endpoint"]["follow_champion"] = setting
    path.write_text(yaml.safe_dump(document))
    values["bundle_mode"] = mode
    enabled = setting
    receipt = {
        "kind": "initial",
        "alias": "champion",
        "model_name": "main.ml.sample_model_alice",
        "new_version": "2",
        "prior_version": None,
    }
    saved = {}
    dbutils = SimpleNamespace(
        jobs=SimpleNamespace(
            taskValues=SimpleNamespace(set=lambda key, value: saved.update({key: value}))
        )
    )
    handoff.publish_serving_handoff(dbutils, receipt, values)
    assert saved == {"serving_requested": enabled, "serving_model_version": "2" if enabled else ""}


@pytest.mark.parametrize(
    "mode,enabled,follow",
    [("development", True, False), ("production", True, False), ("production", False, True)],
)
@pytest.mark.parametrize("bad_binding", ["cost_tag_value", "catalog"])
def test_inactive_handoff_ignores_unused_endpoint_bindings(
    endpoint_project, handoff, mode, enabled, follow, bad_binding
):
    """Training must finish when serving is inactive even if its target setup is incomplete."""
    document, path, values = endpoint_project
    document["serving_endpoint"].update(enabled=enabled, follow_champion=follow)
    path.write_text(yaml.safe_dump(document))
    values.update(bundle_mode=mode)
    values[bad_binding] = (
        "REPLACE_WITH_COST_TAG" if bad_binding == "cost_tag_value" else "${var.catalog}"
    )
    receipt = {
        "kind": "initial",
        "alias": "champion",
        "model_name": "main.ml.sample_model_alice",
        "new_version": "2",
        "prior_version": None,
    }
    saved = {}
    dbutils = SimpleNamespace(
        jobs=SimpleNamespace(
            taskValues=SimpleNamespace(set=lambda key, value: saved.update({key: value}))
        )
    )
    handoff.publish_serving_handoff(dbutils, receipt, values)
    assert saved == {"serving_requested": False, "serving_model_version": ""}


def test_active_handoff_still_requires_valid_endpoint_bindings(endpoint_project, handoff):
    """Moving the optional gate must not dispatch a production endpoint with invalid ownership."""
    _, _, values = endpoint_project
    values.update(bundle_mode="production", cost_tag_value="REPLACE_WITH_COST_TAG")
    dbutils = Mock()
    with pytest.raises(ValueError, match="cost_tag_value"):
        handoff.publish_serving_handoff(dbutils, None, values)
    dbutils.jobs.taskValues.set.assert_not_called()


@pytest.mark.parametrize("flag", ["enabled", "follow_champion"])
@pytest.mark.parametrize("value", ["false", 0, None])
def test_handoff_rejects_malformed_yaml_gate(endpoint_project, handoff, flag, value):
    """A string or integer must not silently turn an automatic serving policy on or off."""
    document, path, values = endpoint_project
    document["serving_endpoint"][flag] = value
    path.write_text(yaml.safe_dump(document))
    values["bundle_mode"] = "production"
    with pytest.raises(ValueError, match=f"{flag} must be a boolean"):
        handoff.publish_serving_handoff(Mock(), None, values)


def test_logging_sinks_cannot_be_shared_with_online_setup(endpoint_project):
    """Different endpoint names still require distinct request payload tables."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    document, path, values = endpoint_project
    logging = {
        "logging_mode": "telemetry",
        "logging_catalog": "main",
        "logging_schema": "ml",
        "logging_table_prefix": "payload",
    }
    document["serving_endpoint"].update(logging)
    document["online_feature_lookup"] = {"endpoint": {"endpoint_name": "other", **logging}}
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="logging"):
        api.read_serving_endpoint_settings(values)


def test_numeric_parity_checks_every_column_and_preserves_order():
    """A wrong status, missing output or reordered prediction must fail smoke."""
    import pandas as pd

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    expected = pd.DataFrame({"prediction": [1.25, None], "status": ["predicted", "excluded"]})
    assert (
        api.verify_serving_predictions(expected, expected.to_dict("records"), channel="REST") == 2
    )
    for actual in (
        [{"prediction": 1.3, "status": "predicted"}, {"prediction": None, "status": "excluded"}],
        [{"prediction": 1.25}, {"prediction": None}],
        [],
        list(reversed(expected.to_dict("records"))),
    ):
        with pytest.raises(ValueError, match="REST"):
            api.verify_serving_predictions(expected, actual, channel="REST")


@pytest.mark.parametrize("channel", ["Reference", "REST", "SQL"])
@pytest.mark.parametrize(
    "dtype,values",
    [
        ("string", [None, None]),
        ("string", ["rule_excluded", None]),
        ("Int64", [9007199254740993, None]),
        ("boolean", [True, None]),
        ("object", ["rule_excluded", None]),
    ],
)
def test_nullable_output_parity_accepts_equivalent_nulls(channel, dtype, values):
    """JSON null must match nullable model outputs without changing their non-null values."""
    import pandas as pd

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    expected = pd.DataFrame({"output": pd.Series(values, dtype=dtype)})
    actual = [{"output": value} for value in values]
    assert api.verify_serving_predictions(expected, actual, channel=channel) == len(values)


@pytest.mark.parametrize("channel", ["Reference", "REST", "SQL"])
@pytest.mark.parametrize(
    "values",
    [
        [None, "rule_excluded"],
        ["different_rule", None],
        ["rule_excluded", ""],
        [None, None],
    ],
)
def test_nullable_string_parity_rejects_changed_values_or_null_positions(channel, values):
    """Normalizing null representations must not hide a moved null or changed exclusion reason."""
    import pandas as pd

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    expected = pd.DataFrame({"reason": pd.Series(["rule_excluded", pd.NA], dtype="string")})
    actual = [{"reason": value} for value in values]
    with pytest.raises(ValueError, match=f"{channel} predictions differ"):
        api.verify_serving_predictions(expected, actual, channel=channel)


def test_empty_smoke_sample_cannot_mutate_cloud():
    """An empty scoring source never earns a successful endpoint smoke check."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    frame = Mock()
    frame.schema = SimpleNamespace(
        fields=[SimpleNamespace(name="value", dataType=SimpleNamespace(typeName=lambda: "double"))]
    )
    frame.selectExpr.return_value = frame
    frame.orderBy.return_value = frame
    frame.limit.return_value = frame
    frame.collect.return_value = []
    frame.__getitem__ = Mock(side_effect=lambda key: key)
    spark = Mock()
    spark.table.return_value = frame
    with pytest.raises(ValueError, match="empty"):
        api.sample_serving_rows(spark, "main.source.input", (("value", "float64"),), rows=20)


@pytest.mark.parametrize("name,quoted", [("a.b", "`a.b`"), ("a`b", "`a``b`")])
def test_rest_only_sampling_escapes_projection_and_sort_names(name, quoted):
    """A literal feature name must not become a nested-field expression during sorting."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    frame = Mock()
    frame.schema = SimpleNamespace(
        fields=[SimpleNamespace(name=name, dataType=SimpleNamespace(typeName=lambda: "double"))]
    )
    frame.selectExpr.return_value = frame
    frame.orderBy.return_value = frame
    frame.limit.return_value = frame
    frame.__getitem__ = Mock(side_effect=lambda key: key)
    frame.collect.return_value = [SimpleNamespace(asDict=lambda **_: {name: 1.0})]
    spark = Mock()
    spark.table.return_value = frame
    assert api.sample_serving_rows(spark, "main.source.input", ((name, "float64"),), rows=1) == [
        {name: 1.0}
    ]
    projection, hashed = frame.selectExpr.call_args_list[0].args
    assert projection == quoted and hashed.startswith(f"xxhash64({quoted}) AS `")
    sort_name = hashed.split(" AS `")[1].removesuffix("`")
    frame.orderBy.assert_called_once_with(sort_name, quoted)
    assert frame.selectExpr.call_args_list[-1].args == (quoted,)


def test_dispatch_submits_once_without_waiting_and_pins_version(handoff):
    """Training finishes after submission; endpoint readiness belongs to the separate child run."""
    client = Mock()
    client.jobs.run_now.return_value.response.run_id = 99
    values = {"serving_job_id": "123", "job_run_id": "456", "model_version": "7"}
    result = handoff.submit_serving_run(client, values)
    assert result["run_id"] == 99 and result["status"] == "SUBMITTED"
    call = client.jobs.run_now.call_args.kwargs
    assert call["job_id"] == 123
    assert call["job_parameters"] == {"model_version": "7"}
    assert len(call["idempotency_token"]) == 64
    token = call["idempotency_token"]
    assert handoff.submit_serving_run(client, values)["run_id"] == 99
    assert client.jobs.run_now.call_args.kwargs["idempotency_token"] == token
    client.jobs.run_now.return_value.result.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [("serving_job_id", ""), ("job_run_id", "{{job.run_id}}"), ("model_version", "champion")],
)
def test_dispatch_validates_identity_before_submitting(handoff, key, value):
    """Unresolved invocation metadata cannot dispatch an unrelated serving job."""
    client = Mock()
    values = {"serving_job_id": "123", "job_run_id": "456", "model_version": "7", key: value}
    with pytest.raises(ValueError, match=key):
        handoff.submit_serving_run(client, values)
    client.jobs.run_now.assert_not_called()


@pytest.fixture
def prepared_job(endpoint_project, monkeypatch):
    """Fake only cloud boundaries while retaining actual job preflight and parity logic."""
    import pandas as pd

    from skyulf.integrations.databricks.serving.contracts import (
        PinnedEndpointPlan,
        PinnedEndpointSpec,
    )

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    _, _, values = endpoint_project
    settings = api.read_serving_endpoint_settings(values)
    spec = PinnedEndpointSpec(
        "sample-endpoint-alice",
        "main.ml.sample_model_alice",
        "2",
        "",
        "",
        "",
        logging_mode="disabled",
    )
    plan = PinnedEndpointPlan(
        spec, {}, ("value",), (("value", "float64"),), (("prediction", "float64"),)
    )
    prepare = Mock(return_value=plan)
    monkeypatch.setattr(api, "prepare_pinned_endpoint", prepare)
    raw = [{"value": 2.0}, {"value": 3.0}]
    monkeypatch.setattr(api, "sample_serving_rows", Mock(return_value=raw))
    predictor = SimpleNamespace(
        predict=lambda frame: pd.DataFrame({"prediction": frame["value"] * 2})
    )
    monkeypatch.setattr(api, "_load_pyfunc", Mock(return_value=predictor))
    events = []
    ensure = Mock(
        side_effect=lambda *args, **kwargs: (
            events.append("endpoint")
            or {"action": "updated", "model_version": "2", "previous_version": "1"}
        )
    )
    monkeypatch.setattr(api, "ensure_owned_endpoint", ensure)
    monkeypatch.setattr(
        api,
        "ensure_owned_sql_function",
        Mock(
            side_effect=lambda *args, **kwargs: (
                events.append("sql")
                or {"action": "unchanged", "function_name": settings["sql_function"]["name"]}
            )
        ),
    )
    monkeypatch.setattr(
        api,
        "grant_serving_access",
        Mock(side_effect=lambda *args, **kwargs: events.append("grants")),
    )
    monkeypatch.setattr(
        api,
        "query_named_records",
        Mock(
            side_effect=lambda *args, **kwargs: (
                events.append("rest")
                or SimpleNamespace(predictions=[{"prediction": 4.0}, {"prediction": 6.0}])
            )
        ),
    )
    monkeypatch.setattr(
        api,
        "_sql_predictions",
        Mock(
            side_effect=lambda *args, **kwargs: (
                events.append("sql_smoke") or [{"prediction": 4.0}, {"prediction": 6.0}]
            )
        ),
    )
    registry = Mock()
    registry.get_model_version_by_alias.return_value.version = "2"
    return api, settings, registry, prepare, ensure, events


@pytest.mark.parametrize("sample_rows", [None, 1, 1000])
def test_job_pins_alias_once_and_compares_both_paths(prepared_job, sample_rows):
    """SQL and REST must compare the same sample against the same registered version."""
    api, settings, registry, prepare, ensure, events = prepared_job
    if sample_rows is not None:
        settings["verification"] = {"sample_rows": sample_rows}
    result = api.run_serving_endpoint(
        None, settings, source_table="main.source.inputs", client=Mock(), registry_client=registry
    )
    assert result["verification"] == {"sampled": 2, "rest_compared": 2, "sql_compared": 2}
    assert "smoke" not in result
    assert api.sample_serving_rows.call_args.kwargs["rows"] == (
        20 if sample_rows is None else sample_rows
    )
    assert events == ["endpoint", "sql", "grants", "rest", "sql_smoke"]
    registry.get_model_version_by_alias.assert_called_once_with(settings["model_name"], "champion")
    assert prepare.call_args.args[0].model_version == "2"
    assert result["payload_table"] is None
    assert ensure.call_args.kwargs["tags"] == {"team": "risk"}
    assert result["examples"]["rest_body"] == {"dataframe_records": [{"value": 2.0}]}


def test_job_refuses_decimal_identity_without_rounding_or_endpoint_writes(
    prepared_job, monkeypatch
):
    """Model-set Decimal keys must fail JSON admission rather than change customer identity."""
    from dataclasses import replace
    from decimal import Decimal

    from test_databricks_decimal_transport import _source, _spark

    api, settings, registry, prepare, ensure, events = prepared_job
    prepare.return_value = replace(
        prepare.return_value,
        input_columns=("id", "value"),
        input_schema=(("id", "object"), ("value", "float64")),
        record_key_columns=("id",),
    )
    source = _source(
        [{"id": Decimal("9007199254740993"), "value": Decimal("2.0")}],
        {"id": "decimal(20,0)", "value": "decimal(17,6)"},
    )
    monkeypatch.setattr(api, "sample_serving_rows", real_sample_serving_rows)
    with pytest.raises(ValueError, match="non-JSON"):
        api.run_serving_endpoint(
            _spark(source),
            settings,
            source_table="main.source.inputs",
            client=Mock(),
            registry_client=registry,
        )
    ensure.assert_not_called()
    assert events == []


def test_generated_python_request_example_runs_without_hidden_variables(prepared_job, monkeypatch):
    """The printed SDK snippet must contain its sample rows so users can run it as copied."""
    from databricks import sdk

    api, settings, registry, _, _, _ = prepared_job
    result = api.run_serving_endpoint(
        None, settings, source_table="main.source.inputs", client=Mock(), registry_client=registry
    )
    caller = Mock()
    monkeypatch.setattr(sdk, "WorkspaceClient", lambda: caller)
    exec(result["examples"]["python"], {})
    caller.serving_endpoints.query.assert_called_once_with(
        name=settings["endpoint_name"], dataframe_records=[{"value": 2.0}]
    )


def test_explicit_version_does_not_read_champion_and_sql_can_be_disabled(prepared_job):
    """Manual version pins stay immutable and REST-only serving needs no SQL privileges."""
    api, settings, registry, prepare, ensure, events = prepared_job
    settings["sql_function"] = {"enabled": False}
    result = api.run_serving_endpoint(
        None,
        settings,
        source_table="main.source.inputs",
        client=Mock(),
        registry_client=registry,
        model_version="2",
    )
    registry.get_model_version_by_alias.assert_not_called()
    assert result["verification"]["sql_compared"] == 0
    assert result["sql_function"] is None
    assert events == ["endpoint", "grants", "rest"]


def test_invalid_verification_fails_before_model_lookup_or_endpoint_write(prepared_job):
    """Direct runtime callers must receive the same preflight bound checks as YAML callers."""
    api, settings, registry, prepare, ensure, events = prepared_job
    settings["verification"] = {"sample_rows": 0}
    with pytest.raises(ValueError, match="verification.sample_rows"):
        api.run_serving_endpoint(
            None,
            settings,
            source_table="main.source.inputs",
            client=Mock(),
            registry_client=registry,
        )
    registry.get_model_version_by_alias.assert_not_called()
    prepare.assert_not_called()
    ensure.assert_not_called()
    assert events == []


@pytest.mark.parametrize("field,value", [("selection", "champion"), ("smoke", {"rows": 1})])
def test_direct_job_rejects_removed_serving_options_before_cloud_work(prepared_job, field, value):
    """Direct calls cannot silently ignore a removed selector or verification size."""
    api, settings, registry, prepare, ensure, events = prepared_job
    settings[field] = value
    with pytest.raises(ValueError, match=f"Unknown serving_endpoint fields.*{field}"):
        api.run_serving_endpoint(
            None,
            settings,
            source_table="main.source.inputs",
            client=Mock(),
            registry_client=registry,
        )
    registry.get_model_version_by_alias.assert_not_called()
    prepare.assert_not_called()
    ensure.assert_not_called()
    assert events == []


@pytest.mark.parametrize("field,value", [("selection", "champion"), ("smoke", {"rows": 1})])
def test_disabled_direct_job_rejects_removed_serving_options(field, value):
    """Disabling an endpoint must not keep removed public option names accepted."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    with pytest.raises(ValueError, match=f"Unknown serving_endpoint fields.*{field}"):
        api.run_serving_endpoint(
            None,
            {"enabled": False, field: value},
            source_table="",
            client=None,
            registry_client=None,
        )


def test_disabled_direct_job_keeps_unused_setting_values_inert():
    """Only option names are checked while disabled, so incomplete endpoint setup stays inert."""
    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    result = api.run_serving_endpoint(
        None,
        {"enabled": False, "verification": {"sample_rows": 0}},
        source_table="",
        client=None,
        registry_client=None,
    )
    assert result == {"status": "DISABLED"}


@pytest.mark.parametrize("invalid", ["permissions", "sql_schema", "empty"])
def test_job_preflight_failure_never_mutates_endpoint(prepared_job, invalid):
    """Bad grants, unsupported SQL inputs and empty samples fail before paid-resource writes."""
    from dataclasses import replace

    api, settings, registry, prepare, ensure, events = prepared_job
    if invalid == "permissions":
        settings["permissions"] = [{"group_name": "users", "level": "CAN_MANAGE"}]
    elif invalid == "sql_schema":
        prepare.return_value = replace(
            prepare.return_value,
            input_columns=("bad name",),
            input_schema=(("bad name", "float64"),),
        )
    else:
        api.sample_serving_rows.side_effect = ValueError("empty sample")
    with pytest.raises(ValueError):
        api.run_serving_endpoint(
            None,
            settings,
            source_table="main.source.inputs",
            client=Mock(),
            registry_client=registry,
        )
    ensure.assert_not_called()
    assert events == []


def test_prediction_mismatch_fails_and_keeps_endpoint_for_inspection(prepared_job):
    """An updated endpoint is never described as verified when inference disagrees."""
    api, settings, registry, prepare, ensure, events = prepared_job
    api.query_named_records.side_effect = None
    api.query_named_records.return_value = SimpleNamespace(
        predictions=[{"prediction": 9.0}, {"prediction": 6.0}]
    )
    with pytest.raises(ValueError, match="REST predictions differ"):
        api.run_serving_endpoint(
            None,
            settings,
            source_table="main.source.inputs",
            client=Mock(),
            registry_client=registry,
        )
    assert ensure.call_count == 1
    assert "sql_smoke" not in events


def test_job_rejects_outer_timeout_shorter_than_readiness(endpoint_project, monkeypatch):
    """The child must have enough wall-clock budget for readiness plus inspection and smoke."""
    from databricks import sdk

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    _, _, values = endpoint_project
    values["job_timeout_s"] = "3600"
    constructor = Mock()
    monkeypatch.setattr(sdk, "WorkspaceClient", constructor)
    with pytest.raises(ValueError, match="ready_timeout_s"):
        api.run_serving_endpoint_notebook(
            None, SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values))
        )
    constructor.assert_not_called()


@pytest.mark.parametrize("expected,actual", [(True, 1), (False, 0), (1, 1.0), (1.25, "1.25")])
def test_parity_rejects_scalar_type_coercion(expected, actual):
    """A transport type change must not be hidden by pandas equality or float conversion."""
    import pandas as pd

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    with pytest.raises(ValueError, match="REST"):
        api.verify_serving_predictions(
            pd.DataFrame({"prediction": [expected]}), [{"prediction": actual}], channel="REST"
        )


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_reference_self_comparison_rejects_infinite_predictions(value):
    """Overflow cannot pass preflight merely because infinity equals itself numerically."""
    import pandas as pd

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    expected = pd.DataFrame({"prediction": [value]})
    with pytest.raises(ValueError, match="Reference"):
        api.verify_serving_predictions(expected, expected.to_dict("records"), channel="Reference")


def test_smoke_codec_works_with_a_fitted_saved_and_reloaded_pyfunc(tmp_path):
    """Use an actual trained MLflow package to prove missing-value transport and complete outputs."""
    import mlflow
    import pandas as pd

    from skyulf import SkyulfPipeline
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
    from skyulf.integrations.databricks.serving.contracts import PinnedEndpointSpec
    from skyulf.integrations.databricks.serving.endpoints import build_pinned_endpoint
    from skyulf.integrations.mlflow.models.pipeline_model import pipeline_model_save_options
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel

    api = import_module("skyulf.integrations.databricks.jobs.serving_endpoint_job")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x"], "strategy": "mean"},
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    artifact_path = tmp_path / "fitted"
    save_pipeline(pipeline, artifact_path)
    artifact = load_pipeline(artifact_path)
    package = tmp_path / "package"
    mlflow.pyfunc.save_model(
        str(package), **pipeline_model_save_options(artifact, artifact_path, tmp_path)
    )
    loaded = mlflow.pyfunc.load_model(str(package))
    spec = PinnedEndpointSpec(
        "model-endpoint", "main.ml.model", "1", "", "", "", logging_mode="disabled"
    )
    resolved = ResolvedModel(
        spec.model_name, "1", spec.model_uri, None, artifact.manifest.pipeline_sha256
    )
    plan = build_pinned_endpoint(
        spec, resolved=resolved, artifact=artifact, package_info=mlflow.models.Model.load(package)
    )
    encoded, records = api._smoke_inputs([{"x": 5.0}, {"x": None}], plan)
    expected = loaded.predict(encoded)
    replay = loaded.predict(pd.DataFrame(records))
    assert isinstance(expected, pd.DataFrame)
    assert isinstance(replay, pd.DataFrame)
    assert api.verify_serving_predictions(expected, replay.to_dict("records"), channel="REST") == 2
    assert expected["prediction"].tolist() == pytest.approx([11.0, 6.0])
    assert records == [{"x": 5.0}, {"x": None}]
    overflow = loaded.predict(pd.DataFrame({"x": [1e308]}))
    assert isinstance(overflow, pd.DataFrame)
    with pytest.raises(ValueError, match="Reference"):
        api.verify_serving_predictions(overflow, overflow.to_dict("records"), channel="Reference")
