"""Two-version routes retain each admitted package and its logging contract."""

from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec

import pytest

pytest.importorskip("mlflow")

from skyulf.integrations.databricks.serving.contracts import (  # noqa: E402
    PinnedEndpointPlan,
    PinnedEndpointSpec,
)
from skyulf.integrations.databricks.serving.endpoints import endpoint_ready  # noqa: E402


@pytest.fixture
def plans():
    """Use already admitted plans to isolate composition from artifact validation."""
    result = []
    for version in ("1", "2"):
        spec = PinnedEndpointSpec(
            "rollout-test", "main.ml.model", version, "main", "obs", "rollout"
        )
        config = {
            "name": spec.endpoint_name,
            "config": {
                "served_entities": [
                    {
                        "entity_name": spec.model_name,
                        "entity_version": version,
                        "workload_type": "CPU",
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True,
                    }
                ]
            },
            "telemetry_config": {
                "table_names": {
                    "logs_table": spec.telemetry_logs_table,
                    "traces_table": spec.telemetry_traces_table,
                    "metrics_table": spec.telemetry_metrics_table,
                },
                "inference_table_config": {"sampling_fraction": 1.0},
                "enabled_telemetry_features": ["TELEMETRY_FEATURE_INFERENCE_TABLE"],
            },
        }
        result.append(
            PinnedEndpointPlan(
                spec, config, ("x",), (("x", "float64"),), (("prediction", "float64"),)
            )
        )
    return tuple(result)


@pytest.fixture
def api():
    """Missing rollout support must fail before exercising route behavior."""
    name = "skyulf.integrations.databricks.serving.rollout_endpoints"
    assert find_spec(name) is not None, "Two-version rollout admission is not implemented"
    return import_module(name)


def ready_endpoint(plan, share=0):
    """Mirror native readback fields independently of the configuration builder."""
    endpoint = deepcopy(plan.config)
    endpoint["state"] = {"ready": "READY", "config_update": "NOT_UPDATING"}
    endpoint["config"]["config_version"] = 3
    endpoint["config"]["traffic_config"] = {
        "routes": [
            {"served_model_name": "champion", "traffic_percentage": 100 - share},
            {"served_model_name": "challenger", "traffic_percentage": share},
        ]
    }
    endpoint["telemetry_config"]["inference_table_config"]["name"] = (
        plan.champion.spec.inference_table
    )
    endpoint["telemetry_config"]["table_names"] = {
        "logs_table": plan.champion.spec.telemetry_logs_table
    }
    return endpoint


def test_two_version_plan_preserves_inputs_and_never_mutates_pinned_plan(api, plans):
    """Daily A/B starts at zero exposure while preserving both exact packages."""
    before = deepcopy(plans)
    plan = api.build_rollout_endpoint(*plans)
    assert plans == before
    assert [(e["name"], e["entity_version"]) for e in plan.config["config"]["served_entities"]] == [
        ("champion", "1"),
        ("challenger", "2"),
    ]
    assert plan.config["config"]["traffic_config"]["routes"] == [
        {"served_model_name": "champion", "traffic_percentage": 100},
        {"served_model_name": "challenger", "traffic_percentage": 0},
    ]
    assert api.rollout_endpoint_ready(ready_endpoint(plan), plan, challenger_percentage=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_schema", (("x", "string"),)),
        ("input_columns", ("y",)),
        ("output_schema", (("prediction", "int64"),)),
    ],
)
def test_incompatible_package_contract_rejected(api, plans, field, value):
    """Traffic cannot switch between different request or response meanings."""
    with pytest.raises(ValueError, match="schema|columns"):
        api.build_rollout_endpoint(plans[0], replace(plans[1], **{field: value}))


def test_same_model_version_and_foreign_endpoint_rejected(api, plans):
    """A rollout requires distinct concrete releases at one explicitly selected endpoint."""
    with pytest.raises(ValueError, match="distinct"):
        api.build_rollout_endpoint(plans[0], plans[0])
    other = replace(plans[1], spec=replace(plans[1].spec, endpoint_name="different"))
    with pytest.raises(ValueError, match="endpoint"):
        api.build_rollout_endpoint(plans[0], other)


@pytest.mark.parametrize("share", [True, -1, 101, 0.5, "10"])
def test_invalid_share_rejected_before_remote_work(api, plans, share):
    """Route percentages must be exact bounded integer percentage points."""
    plan = api.build_rollout_endpoint(*plans)
    with pytest.raises(ValueError, match="percentage"):
        api.rollout_endpoint_ready(ready_endpoint(plan), plan, challenger_percentage=share)


def test_readback_rejects_unknown_route_wrong_version_and_logging(api, plans):
    """A healthy state flag cannot conceal a changed route, model or telemetry sink."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan, 20)
    assert api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=20)
    for section, key, value in [
        ("entity", "entity_version", "99"),
        ("route", "served_model_name", "foreign"),
        ("logging", "sampling_fraction", 0.5),
    ]:
        changed = deepcopy(endpoint)
        target = {
            "entity": changed["config"]["served_entities"][1],
            "route": changed["config"]["traffic_config"]["routes"][1],
            "logging": changed["telemetry_config"]["inference_table_config"],
        }[section]
        target[key] = value
        with pytest.raises(ValueError):
            api.rollout_endpoint_ready(changed, plan, challenger_percentage=20)


def test_pending_updates_are_not_ready_and_pinned_endpoint_stays_strict(api, plans):
    """A pending deployment cannot be advanced and A/B cannot impersonate pinned SQL."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan)
    with pytest.raises(ValueError):
        endpoint_ready(endpoint, plans[0])
    endpoint["state"]["config_update"] = "IN_PROGRESS"
    assert not api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


def test_mutated_plan_cannot_redirect_creation_to_a_foreign_endpoint(api, plans):
    """Matching mutated requests must still agree with the inspected endpoint selectors."""
    for plan in plans:
        plan.config["name"] = "foreign-endpoint"
    with pytest.raises(ValueError, match="endpoint"):
        api.build_rollout_endpoint(*plans)


def test_unadmitted_environment_changes_fail_at_build_and_readback(api, plans):
    """Model configuration environment variables can alter behavior without changing version."""
    plan = api.build_rollout_endpoint(*plans)
    changed = ready_endpoint(plan)
    changed["config"]["served_entities"][1]["environment_vars"] = {"MODEL_CONFIG": "different"}
    with pytest.raises(ValueError, match="configuration"):
        api.rollout_endpoint_ready(changed, plan, challenger_percentage=0)
    plans[1].config["config"]["served_entities"][0]["environment_vars"] = {
        "MODEL_CONFIG": "different"
    }
    with pytest.raises(ValueError, match="configuration"):
        api.build_rollout_endpoint(*plans)


@pytest.mark.parametrize("field", ["logs_table", "traces", "metrics", "sampling_fraction"])
def test_changed_active_telemetry_contract_rejected(api, plans, field):
    """Only unused CREATE sink placeholders may disappear from inference-only readback."""
    plan = api.build_rollout_endpoint(*plans)
    changed = ready_endpoint(plan)
    telemetry = changed["telemetry_config"]
    if field == "logs_table":
        telemetry["table_names"]["logs_table"] = "foreign.logs.changed_sink"
    elif field in {"traces", "metrics"}:
        telemetry["enabled_telemetry_features"].append("TELEMETRY_FEATURE_" + field.upper())
    else:
        telemetry["inference_table_config"]["sampling_fraction"] = 0.5
    with pytest.raises(ValueError, match="config"):
        api.rollout_endpoint_ready(changed, plan, challenger_percentage=0)


def test_online_lookup_contract_must_match_between_versions(api, plans):
    """Identical schemas cannot conceal different feature freshness or lookup meaning."""
    first = replace(plans[0], online_contract="a" * 64)
    second = replace(plans[1], online_contract="b" * 64)
    with pytest.raises(ValueError, match="online_contract"):
        api.build_rollout_endpoint(first, second)
    with pytest.raises(ValueError, match="online_contract"):
        api.build_rollout_endpoint(first, plans[1])
    assert api.build_rollout_endpoint(first, replace(second, online_contract="a" * 64))


def test_unadmitted_root_option_rejected(api, plans):
    """Both plans agreeing cannot authorize an uninspected endpoint optimization."""
    for plan in plans:
        plan.config["route_optimized"] = True
    with pytest.raises(ValueError, match="configuration"):
        api.build_rollout_endpoint(*plans)


def test_legacy_parallel_entities_rejected(api, plans):
    """An additional legacy deployment list is not part of the two-version plan."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan)
    endpoint["config"]["served_models"] = [{"name": "unadmitted"}]
    with pytest.raises(ValueError, match="configuration"):
        api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


def native_endpoint(plan):
    """Preserve the real GET defaults and equivalent deprecated representation."""
    endpoint = ready_endpoint(plan)
    for entity in endpoint["config"]["served_entities"]:
        entity.update(
            type="UC_MODEL",
            batching_config={"enabled": False},
            state={"deployment": "DEPLOYMENT_READY"},
            creator="fixture@example.com",
            creation_timestamp=123456789,
        )
    aliases = deepcopy(endpoint["config"]["served_entities"])
    for entity in aliases:
        entity["model_name"] = entity.pop("entity_name")
        entity["model_version"] = entity.pop("entity_version")
    endpoint["config"]["served_models"] = list(reversed(aliases))
    return endpoint


def test_native_readback_accepts_equivalent_aliases_and_disabled_batching(api, plans):
    """Native GET metadata must not block an otherwise identical admitted rollout."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = native_endpoint(plan)
    before = deepcopy(endpoint)
    assert api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)
    assert endpoint == before
    with pytest.raises(ValueError):
        endpoint_ready(endpoint, plans[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "foreign.models.other"),
        ("model_version", "99"),
        ("state", {"deployment": "DEPLOYMENT_FAILED"}),
        ("environment_vars", {"MODEL_CONFIG": "foreign"}),
        ("batching_config", {"enabled": True}),
        ("batching_config", {"enabled": 0}),
        ("entity_name", "foreign.models.duplicate_alias"),
    ],
)
def test_native_legacy_alias_must_match_entire_entity(api, plans, field, value):
    """Legacy fields cannot conceal contradictory versions, metadata or runtime options."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = native_endpoint(plan)
    endpoint["config"]["served_models"][0][field] = value
    with pytest.raises(ValueError, match="configuration"):
        api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


@pytest.mark.parametrize("aliases", ["duplicate", "missing", "malformed"])
def test_native_alias_membership_is_exact(api, plans, aliases):
    """An equivalent legacy list must cover both named entities exactly once."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = native_endpoint(plan)
    rows = endpoint["config"]["served_models"]
    endpoint["config"]["served_models"] = {
        "duplicate": [rows[0], rows[0]],
        "missing": rows[:1],
        "malformed": {"entities": rows},
    }[aliases]
    with pytest.raises(ValueError, match="configuration"):
        api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "EXTERNAL_MODEL"),
        ("type", None),
        ("batching_config", {"enabled": True}),
        ("batching_config", {"enabled": 0}),
        ("batching_config", {"enabled": False, "max_batch_size": 32}),
        ("environment_vars", {"MODEL_CONFIG": "foreign"}),
    ],
)
def test_matching_native_aliases_cannot_admit_different_runtime(api, plans, field, value):
    """Agreement between both native views is insufficient to change admitted execution."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = native_endpoint(plan)
    endpoint["config"]["served_entities"][0][field] = value
    endpoint["config"]["served_models"][1][field] = value
    with pytest.raises(ValueError, match="configuration"):
        api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


@pytest.mark.parametrize("share", [0, 50, 100])
def test_native_route_name_aliases_agree_without_mutating_readback(api, plans, share):
    """Native routing exposes both names for the same admitted served entity."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan, share)
    for route in endpoint["config"]["traffic_config"]["routes"]:
        route["served_entity_name"] = route["served_model_name"]
    before = deepcopy(endpoint)
    assert api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=share)
    assert endpoint == before


@pytest.mark.parametrize(
    "change",
    ["contradictory", "unknown_field", "bool_share", "duplicate", "missing_name", "unknown_entity"],
)
def test_native_route_aliases_preserve_exact_route_admission(api, plans, change):
    """Equivalent names cannot authorize extra settings, duplicate entities or bool shares."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan)
    routes = endpoint["config"]["traffic_config"]["routes"]
    for route in routes:
        route["served_entity_name"] = route["served_model_name"]
    if change == "contradictory":
        routes[0]["served_entity_name"] = "challenger"
    elif change == "unknown_field":
        routes[0]["unadmitted"] = False
    elif change == "bool_share":
        routes[1]["traffic_percentage"] = False
    elif change == "duplicate":
        routes[1] = deepcopy(routes[0])
    elif change == "missing_name":
        routes[0].pop("served_model_name")
    else:
        routes[0].update(served_entity_name="foreign", served_model_name="foreign")
    with pytest.raises(ValueError, match="traffic"):
        api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


@pytest.mark.parametrize("stamp", [0, 1791662862000, 253402300799999])
def test_native_config_start_time_is_readback_metadata(api, plans, stamp):
    """A native pending-config epoch timestamp does not change the deployment contract."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan)
    endpoint["config"]["start_time"] = stamp
    assert api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)


@pytest.mark.parametrize("stamp", [True, -1, 1.5, "1791662862000", None, 253402300800000])
def test_native_config_start_time_requires_bounded_integer_milliseconds(api, plans, stamp):
    """The metadata allowance cannot accept malformed values or arbitrary configuration."""
    plan = api.build_rollout_endpoint(*plans)
    endpoint = ready_endpoint(plan)
    endpoint["config"]["start_time"] = stamp
    with pytest.raises(ValueError, match="configuration"):
        api.rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)
