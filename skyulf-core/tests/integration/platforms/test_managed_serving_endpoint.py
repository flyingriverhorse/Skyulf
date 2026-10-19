"""Managed serving mutates only the explicitly owned single-model surface."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("databricks.sdk")

from databricks.sdk.errors import NotFound  # noqa: E402
from databricks.sdk.service import serving  # noqa: E402

from skyulf.integrations.databricks.serving import endpoints  # noqa: E402
from skyulf.integrations.databricks.serving import managed_endpoint as managed  # noqa: E402
from skyulf.integrations.databricks.serving.contracts import (  # noqa: E402
    PinnedEndpointPlan,
    PinnedEndpointSpec,
)
from skyulf.integrations.databricks.serving.sql_functions import (  # noqa: E402
    build_serving_sql_function,
)


@pytest.fixture
def ownership():
    """Every target has an independently owned endpoint."""
    return {
        "project": "risk",
        "target": "prod",
        "suffix": "prod",
        "managed_by": "serving_endpoint",
    }


@pytest.fixture
def plan():
    """The prepared package carries both wire and SQL input contracts."""
    spec = PinnedEndpointSpec("risk-prod", "main.ml.risk", "8", "", "", "", logging_mode="disabled")
    entity = {
        "entity_name": spec.model_name,
        "entity_version": "8",
        "workload_type": "CPU",
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    }
    return PinnedEndpointPlan(
        spec,
        {"name": spec.endpoint_name, "config": {"served_entities": [entity]}},
        ("x",),
        (("x", "float64"),),
        (("prediction", "float64"),),
    )


def remote(plan, ownership, version="8"):
    """Readback includes the entity name, exact route and settled service state."""
    entity = dict(
        plan.config["config"]["served_entities"][0], name="risk-prod", entity_version=version
    )
    return {
        "name": "risk-prod",
        "id": "endpoint-id",
        "tags": [{"key": k, "value": v} for k, v in ownership.items()],
        "state": {"ready": "READY", "config_update": "NOT_UPDATING"},
        "config": {
            "served_entities": [entity],
            "traffic_config": {
                "routes": [{"served_entity_name": "risk-prod", "traffic_percentage": 100}]
            },
        },
    }


def test_create_tags_are_atomic_and_plan_is_unchanged(plan, ownership):
    """A failed tag patch must never leave an unowned paid endpoint behind."""
    client = Mock()
    client.serving_endpoints.get.side_effect = [
        NotFound("absent"),
        NotFound("absent"),
        remote(plan, ownership),
    ]
    before = deepcopy(plan.config)
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    kwargs = client.serving_endpoints.create.call_args.kwargs
    assert {tag.key: tag.value for tag in kwargs["tags"]} == ownership
    assert kwargs["config"].served_entities[0].name == "risk-prod"
    assert kwargs["config"].traffic_config.routes[0].traffic_percentage == 100
    assert result["action"] == "created" and result["previous_version"] is None
    assert plan.config == before


def test_owned_update_pins_version_and_keeps_entity_name(plan, ownership):
    """Champion changes update the same owned name instead of adding a route."""
    client = Mock()
    client.serving_endpoints.get.side_effect = [
        remote(plan, ownership, "7"),
        remote(plan, ownership),
    ]
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    kwargs = client.serving_endpoints.update_config.call_args.kwargs
    assert kwargs["served_entities"][0].entity_version == "8"
    assert kwargs["served_entities"][0].name == "risk-prod"
    assert result["action"] == "updated" and result["previous_version"] == "7"
    client.serving_endpoints.create.assert_not_called()


def test_identical_owned_endpoint_is_unchanged(plan, ownership):
    """Reruns do not restart an already settled endpoint."""
    client = Mock()
    client.serving_endpoints.get.return_value = remote(plan, ownership)
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    assert result["action"] == "unchanged"
    client.serving_endpoints.update_config.assert_not_called()


@pytest.mark.parametrize("operation", ["create", "update"])
def test_sdk_serializes_named_route_in_request_body(plan, ownership, operation, monkeypatch):
    """SDK serialization must retain route identity for runtime-provided older SDKs."""
    transport = Mock()
    transport.do.return_value = {"name": plan.spec.endpoint_name}
    api = serving.ServingEndpointsAPI(transport)
    reads = (
        [NotFound("absent"), NotFound("absent"), remote(plan, ownership)]
        if operation == "create"
        else [remote(plan, ownership, "7"), remote(plan, ownership)]
    )
    monkeypatch.setattr(api, "get", Mock(side_effect=reads))
    result = managed.ensure_owned_endpoint(
        SimpleNamespace(serving_endpoints=api), plan, ownership=ownership
    )
    request = transport.do.call_args.kwargs
    body = request["body"]
    config = body["config"] if operation == "create" else body
    assert transport.do.call_args.args[0] == ("POST" if operation == "create" else "PUT")
    assert config["traffic_config"]["routes"] == [
        {"served_model_name": "risk-prod", "traffic_percentage": 100}
    ]
    assert config["served_entities"][0]["name"] == "risk-prod"
    assert result["action"] == ("created" if operation == "create" else "updated")


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "pending",
        "in_progress",
        "logging",
        "multiple",
        "traffic",
        "foreign_model",
        "unnamed",
    ],
)
def test_refuses_conflicts_without_writes(plan, ownership, change):
    """Ownership, topology and logging are admission checks before any mutation."""
    endpoint = remote(plan, ownership, "7")
    if change == "owner":
        endpoint["tags"][0]["value"] = "foreign"
    elif change == "pending":
        endpoint["pending_config"] = {"config_version": 2}
    elif change == "in_progress":
        endpoint["state"]["config_update"] = "IN_PROGRESS"
    elif change == "logging":
        endpoint["ai_gateway"] = {"inference_table_config": {"enabled": True}}
    elif change == "multiple":
        endpoint["config"]["served_entities"] *= 2
    elif change == "traffic":
        endpoint["config"]["traffic_config"]["routes"][0]["traffic_percentage"] = 50
    elif change == "foreign_model":
        endpoint["config"]["served_entities"][0]["entity_name"] = "main.ml.foreign"
    else:
        del endpoint["config"]["served_entities"][0]["name"]
    client = Mock()
    client.serving_endpoints.get.return_value = endpoint
    with pytest.raises((ValueError, RuntimeError)):
        managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    client.serving_endpoints.update_config.assert_not_called()
    client.serving_endpoints.create.assert_not_called()


def test_managed_sql_is_stable_across_model_versions_and_replaced_for_wire_changes(plan, ownership):
    """Stable endpoint SQL survives version updates but refreshes its transport codec."""
    client, spark = Mock(), Mock()
    client.serving_endpoints.get.return_value = remote(plan, ownership)
    client.functions.get.side_effect = NotFound("absent")
    function = build_serving_sql_function(plan, "main.ml.score")
    result = managed.ensure_owned_sql_function(spark, client, function, ownership=ownership)
    ddl = spark.sql.call_args.args[0]
    comment = ddl.split("COMMENT '")[1].split("'\n")[0]
    assert result["action"] == "created"
    assert "models:/main.ml.risk/8" not in comment
    client.functions.get.side_effect = None
    client.functions.get.return_value = {"comment": comment}
    spark.reset_mock()
    next_config = deepcopy(plan.config)
    next_config["config"]["served_entities"][0]["entity_version"] = "9"
    changed_version = replace(plan, spec=replace(plan.spec, model_version="9"), config=next_config)
    client.serving_endpoints.get.return_value = remote(changed_version, ownership, "9")
    result = managed.ensure_owned_sql_function(
        spark,
        client,
        build_serving_sql_function(changed_version, "main.ml.score"),
        ownership=ownership,
    )
    assert result["action"] == "unchanged"
    spark.sql.assert_not_called()
    encoded = replace(changed_version, input_schema=(("x", "Int64"),))
    result = managed.ensure_owned_sql_function(
        spark, client, build_serving_sql_function(encoded, "main.ml.score"), ownership=ownership
    )
    assert result["action"] == "updated"
    assert spark.sql.call_args.args[0].startswith("CREATE OR REPLACE FUNCTION")
    assert "CAST(`x` AS STRING)" in spark.sql.call_args.args[0]


def test_foreign_function_is_never_replaced(plan, ownership):
    """A shared UC schema must not let deployment erase unrelated functions."""
    client, spark = Mock(), Mock()
    client.functions.get.return_value = {"comment": "someone else's function"}
    with pytest.raises(ValueError, match="owned"):
        managed.ensure_owned_sql_function(
            spark, client, build_serving_sql_function(plan, "main.ml.score"), ownership=ownership
        )
    spark.sql.assert_not_called()


def test_additive_grants_preserve_acls_and_escape_principals(plan):
    """Adding callers never replaces permissions or interpolates principal SQL."""
    client, spark = Mock(), Mock()
    client.serving_endpoints.get.return_value = {"id": "endpoint-id"}
    managed.grant_serving_access(
        spark,
        client,
        plan,
        principals=[{"group_name": "data`science", "level": "CAN_QUERY"}],
        function_name="main.ml.score",
    )
    request = client.serving_endpoints.update_permissions.call_args
    assert request.args == ("endpoint-id",)
    assert request.kwargs["access_control_list"][0].group_name == "data`science"
    assert spark.sql.call_args.args[0] == (
        "GRANT EXECUTE ON FUNCTION `main`.`ml`.`score` TO `data``science`"
    )
    client.serving_endpoints.set_permissions.assert_not_called()


@pytest.mark.parametrize(
    "principals",
    [
        [{"group_name": ""}],
        [{"group_name": "x", "user_name": "y"}],
        [{"group_name": "x", "level": "CAN_MANAGE"}],
        "group",
    ],
)
def test_principals_fail_locally(principals):
    """Malformed ACL configuration must fail before endpoint creation or grants."""
    with pytest.raises(ValueError):
        managed.validate_serving_principals(principals)


def test_nullable_wire_change_replaces_identical_sql_types(plan, ownership):
    """Int64 and int64 share SQL BIGINT but require different request encodings."""
    client, spark = Mock(), Mock()
    client.serving_endpoints.get.return_value = remote(plan, ownership)
    client.functions.get.side_effect = NotFound("absent")
    original = build_serving_sql_function(
        replace(plan, input_schema=(("x", "int64"),)), "main.ml.score"
    )
    managed.ensure_owned_sql_function(spark, client, original, ownership=ownership)
    ddl = spark.sql.call_args.args[0]
    comment = ddl.split("COMMENT '")[1].split("'\n")[0]
    client.functions.get.side_effect = None
    client.functions.get.return_value = {"comment": comment}
    encoded = build_serving_sql_function(
        replace(plan, input_schema=(("x", "Int64"),)), "main.ml.score"
    )
    result = managed.ensure_owned_sql_function(spark, client, encoded, ownership=ownership)
    assert original.response_type == encoded.response_type
    assert result["action"] == "updated"
    assert "CAST(`x` AS STRING)" in spark.sql.call_args.args[0]


def test_sql_refuses_foreign_endpoint_before_ddl(plan, ownership):
    """A ready model at the right name does not establish endpoint ownership."""
    client, spark = Mock(), Mock()
    endpoint = remote(plan, ownership)
    endpoint["tags"] = []
    client.serving_endpoints.get.return_value = endpoint
    client.functions.get.side_effect = NotFound("absent")
    with pytest.raises(ValueError, match="owned"):
        managed.ensure_owned_sql_function(
            spark, client, build_serving_sql_function(plan, "main.ml.score"), ownership=ownership
        )
    spark.sql.assert_not_called()


def test_disabled_plan_rejects_legacy_payload_logging(plan, ownership):
    """Legacy inference tables must not bypass disabled payload logging admission."""
    endpoint = remote(plan, ownership, "7")
    endpoint["config"]["auto_capture_config"] = {"enabled": True}
    client = Mock()
    client.serving_endpoints.get.return_value = endpoint
    with pytest.raises(ValueError, match="logging|settings"):
        managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    client.serving_endpoints.update_config.assert_not_called()


def test_ready_readback_rejects_changed_route(plan, ownership):
    """READY alone cannot admit traffic changed by a concurrent writer."""
    client = Mock()
    changed = remote(plan, ownership)
    changed["config"]["traffic_config"]["routes"][0]["traffic_percentage"] = 50
    client.serving_endpoints.get.side_effect = [remote(plan, ownership, "7"), changed]
    with pytest.raises(ValueError, match="traffic"):
        managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    assert client.serving_endpoints.update_config.call_count == 1


def test_telemetry_create_includes_atomic_tags(plan, ownership):
    """Native telemetry transport retains ownership in the initial POST body."""
    spec = replace(
        plan.spec,
        logging_mode="telemetry",
        logging_catalog="main",
        logging_schema="logs",
        logging_table_prefix="risk",
    )
    telemetry = {
        "table_names": {"logs_table": spec.telemetry_logs_table},
        "inference_table_config": {"name": spec.inference_table, "sampling_fraction": 1.0},
        "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
    }
    config = deepcopy(plan.config)
    config["telemetry_config"] = deepcopy(telemetry)
    telemetry_plan = replace(plan, spec=spec, config=config)
    endpoint = remote(telemetry_plan, ownership)
    endpoint["telemetry_config"] = telemetry
    endpoint["tags"].append({"key": "cost_center", "value": "ml-platform"})
    client = Mock()
    client.api_client.do.side_effect = [NotFound("absent"), {}, endpoint]
    client.serving_endpoints.get.side_effect = NotFound("absent")
    result = managed.ensure_owned_endpoint(
        client, telemetry_plan, ownership=ownership, tags={"cost_center": "ml-platform"}
    )
    post = client.api_client.do.call_args_list[1].kwargs
    assert post["method"] == "POST"
    assert {tag["key"]: tag["value"] for tag in post["body"]["tags"]} == (
        ownership | {"cost_center": "ml-platform"}
    )
    assert result["action"] == "created"
    client.serving_endpoints.create.assert_not_called()


def test_managed_update_waits_for_both_settled_states(plan, ownership, monkeypatch):
    """READY during a swap still serves the prior version until NOT_UPDATING."""
    pending = remote(plan, ownership, "7")
    pending["state"]["config_update"] = "IN_PROGRESS"
    client = Mock()
    client.serving_endpoints.get.side_effect = [
        remote(plan, ownership, "7"),
        pending,
        remote(plan, ownership),
    ]
    monkeypatch.setattr(endpoints, "sleep", lambda seconds: None)
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    assert result["action"] == "updated"
    assert client.serving_endpoints.get.call_count == 3


def test_native_disabled_batching_is_admitted(plan, ownership):
    """Databricks readback may include explicit platform defaults on native entities."""
    endpoint = remote(plan, ownership)
    endpoint["config"]["served_entities"][0]["batching_config"] = {"enabled": False}
    client = Mock()
    client.serving_endpoints.get.return_value = endpoint
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    assert result["action"] == "unchanged"


@pytest.mark.parametrize("owner", [{}, {"managed_by": "other"}, None])
def test_invalid_ownership_fails_before_api(plan, owner):
    """Missing ownership configuration cannot create or adopt cloud resources."""
    client = Mock()
    with pytest.raises(ValueError, match="ownership"):
        managed.ensure_owned_endpoint(client, plan, ownership=owner)
    assert client.mock_calls == []


@pytest.mark.parametrize("version", [None, "champion", "0", "-1", "1.2", 7])
def test_existing_version_must_be_a_concrete_positive_string(plan, ownership, version):
    """A malformed prior release cannot be adopted as managed endpoint history."""
    client = Mock()
    client.serving_endpoints.get.return_value = remote(plan, ownership, version)
    with pytest.raises(ValueError, match="model_version"):
        managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    client.serving_endpoints.update_config.assert_not_called()


def recorded_endpoint(plan, ownership, version):
    """Sanitize native-endpoint-current.json recorded GET from 2026-10-16."""
    endpoint = remote(plan, ownership, version)
    config = endpoint["config"]
    config["config_version"] = 1
    entity = config["served_entities"][0]
    entity.update(
        creation_timestamp=1792172027000,
        creator="builder@example.invalid",
        state={"deployment": "DEPLOYMENT_READY", "deployment_state_message": ""},
    )
    config["served_models"] = [
        {
            {"entity_name": "model_name", "entity_version": "model_version"}.get(key, key): value
            for key, value in entity.items()
        }
    ]
    config["traffic_config"]["routes"][0]["served_model_name"] = entity["name"]
    endpoint.update(
        creation_timestamp=1792172027000,
        last_updated_timestamp=1792172027000,
        creator="builder@example.invalid",
        description="",
        permission_level="CAN_MANAGE",
        route_optimized=False,
    )
    return endpoint


@pytest.mark.parametrize(
    "logging_mode,sdk_serialized",
    [
        ("disabled", False),
        ("disabled", True),
        ("telemetry", False),
    ],
)
def test_update_accepts_recorded_native_get_shape(plan, ownership, logging_mode, sdk_serialized):
    """Both raw GET and SDK aliases must support a real single-model version update."""
    from databricks.sdk.service.serving import ServingEndpointDetailed

    old, current = recorded_endpoint(plan, ownership, "7"), recorded_endpoint(plan, ownership, "8")
    if logging_mode == "telemetry":
        spec = replace(
            plan.spec,
            logging_mode="telemetry",
            logging_catalog="main",
            logging_schema="logs",
            logging_table_prefix="risk",
        )
        telemetry = {
            "table_names": {"logs_table": spec.telemetry_logs_table},
            "inference_table_config": {"name": spec.inference_table, "sampling_fraction": 1.0},
            "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
        }
        plan = replace(plan, spec=spec, config=plan.config | {"telemetry_config": telemetry})
        old["telemetry_config"] = current["telemetry_config"] = telemetry
    if sdk_serialized:
        old, current = (ServingEndpointDetailed.from_dict(value) for value in (old, current))
    client = Mock()
    if logging_mode == "telemetry":
        client.api_client.do.side_effect = [old, current]
    else:
        client.serving_endpoints.get.side_effect = [old, current]
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    assert result["previous_version"] == "7"
    assert result["action"] == "updated"


@pytest.mark.parametrize("sdk_serialized", [False, True])
def test_explicit_inactive_sdk_defaults_are_admitted(plan, ownership, sdk_serialized):
    """A default-valued API field does not turn a valid managed endpoint foreign."""
    from databricks.sdk.service.serving import ServingEndpointDetailed

    endpoint = remote(plan, ownership)
    endpoint["config"]["auto_capture_config"] = {"enabled": False}
    entity = endpoint["config"]["served_entities"][0]
    entity.update(
        min_provisioned_throughput=0,
        max_provisioned_throughput=0,
        provisioned_model_units=0,
        min_provisioned_concurrency=0,
        max_provisioned_concurrency=0,
        environment_vars={},
    )
    if sdk_serialized:
        endpoint = ServingEndpointDetailed.from_dict(endpoint)
    client = Mock()
    client.serving_endpoints.get.return_value = endpoint
    result = managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    assert result["action"] == "unchanged"
    client.serving_endpoints.update_config.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"min_provisioned_throughput": 1},
        {"max_provisioned_throughput": 1},
        {"provisioned_model_units": 1},
        {"min_provisioned_concurrency": 1},
        {"max_provisioned_concurrency": 1},
        {"environment_vars": {"TOKEN": "secret-ref"}},
        {"burst_scaling_enabled": False},
        {"unknown_feature_enabled": False},
    ],
)
def test_active_or_unknown_runtime_extras_are_refused(plan, ownership, extra):
    """Only named inactive defaults may be omitted during a version replacement."""
    endpoint = remote(plan, ownership, "7")
    endpoint["config"]["served_entities"][0].update(extra)
    client = Mock()
    client.serving_endpoints.get.return_value = endpoint
    with pytest.raises(ValueError, match="settings"):
        managed.ensure_owned_endpoint(client, plan, ownership=ownership)
    client.serving_endpoints.update_config.assert_not_called()


def test_cost_tags_are_atomic_on_create(plan, ownership):
    """The billable endpoint must carry cost attribution in its very first request."""
    client = Mock()
    current = remote(plan, ownership)
    current["tags"].append({"key": "cost_center", "value": "ml-platform"})
    client.serving_endpoints.get.side_effect = [NotFound("absent"), NotFound("absent"), current]
    result = managed.ensure_owned_endpoint(
        client, plan, ownership=ownership, tags={"cost_center": "ml-platform"}
    )
    tags = client.serving_endpoints.create.call_args.kwargs["tags"]
    assert {tag.key: tag.value for tag in tags} == ownership | {"cost_center": "ml-platform"}
    assert result["action"] == "created"
    client.serving_endpoints.patch.assert_not_called()


@pytest.mark.parametrize("prior", [None, "old-budget"])
def test_cost_tags_reconcile_additively_without_restarting_model(plan, ownership, prior):
    """Missing or changed cost attribution is repaired without dropping unrelated tags."""
    initial = remote(plan, ownership)
    initial["tags"].append({"key": "organization", "value": "analytics"})
    if prior is not None:
        initial["tags"].append({"key": "cost_center", "value": prior})
    current = deepcopy(initial)
    current["tags"] = [tag for tag in current["tags"] if tag["key"] != "cost_center"]
    current["tags"].append({"key": "cost_center", "value": "ml-platform"})
    client = Mock()
    client.serving_endpoints.get.side_effect = [initial, current]
    result = managed.ensure_owned_endpoint(
        client, plan, ownership=ownership, tags={"cost_center": "ml-platform"}
    )
    patch = client.serving_endpoints.patch.call_args
    assert patch.args == (plan.spec.endpoint_name,)
    assert {tag.key: tag.value for tag in patch.kwargs["add_tags"]} == {
        "cost_center": "ml-platform"
    }
    assert "delete_tags" not in patch.kwargs
    assert result["action"] == "updated" and result["tags_updated"] is True
    client.serving_endpoints.update_config.assert_not_called()


def test_matching_cost_tags_are_idempotent(plan, ownership):
    """A routine rerun does not mutate matching attribution or other endpoint tags."""
    endpoint = remote(plan, ownership)
    endpoint["tags"].append({"key": "cost_center", "value": "ml-platform"})
    client = Mock()
    client.serving_endpoints.get.return_value = endpoint
    result = managed.ensure_owned_endpoint(
        client, plan, ownership=ownership, tags={"cost_center": "ml-platform"}
    )
    assert result["action"] == "unchanged" and result["tags_updated"] is False
    client.serving_endpoints.patch.assert_not_called()


def test_cost_tag_patch_requires_exact_readback(plan, ownership):
    """A tag API that silently ignores attribution cannot yield a successful result."""
    client = Mock()
    client.serving_endpoints.get.return_value = remote(plan, ownership)
    with pytest.raises(ValueError, match="extra tags"):
        managed.ensure_owned_endpoint(
            client, plan, ownership=ownership, tags={"cost_center": "ml-platform"}
        )
    assert client.serving_endpoints.patch.call_count == 1


@pytest.mark.parametrize(
    "tags",
    [
        {"project": "override"},
        {"managed_by": "serving_endpoint"},
        {"": "value"},
        {"cost": ""},
        {"cost": None},
        {"x" * 129: "value"},
        {"cost": "x" * 257},
        [],
    ],
)
def test_invalid_cost_tags_fail_before_remote_reads(plan, ownership, tags):
    """Cost configuration cannot override ownership or fail after resource creation."""
    client = Mock()
    with pytest.raises(ValueError, match="tags"):
        managed.ensure_owned_endpoint(client, plan, ownership=ownership, tags=tags)
    assert client.mock_calls == []


def test_tag_preflight_is_public_idempotent_and_copies_input():
    """Job settings can validate attribution before any endpoint preparation or mutation."""
    original = {"cost_center": "ml-platform"}
    validated = managed.validate_serving_tags(original)
    assert validated == original and validated is not original
    assert managed.validate_serving_tags(validated) == validated
    assert managed.validate_serving_tags(None) == {}
