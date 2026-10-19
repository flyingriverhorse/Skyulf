"""Explicit ownership admission for stable Databricks endpoint and SQL names."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .contracts import PinnedEndpointPlan
from .endpoints import (
    create_pinned_endpoint,
    read_pinned_endpoint,
    require_pinned_endpoint_logging,
    require_pinned_endpoint_ready,
    wait_for_pinned_endpoint_ready,
)
from .sql_functions import ServingSQLFunctionPlan, build_serving_sql_function

_OWNER_KEYS = {"project", "target", "suffix", "managed_by"}
_COMMENT_PREFIX = "Managed serving function "
_PRINCIPAL_KEYS = {"user_name", "group_name", "service_principal_name"}
_ZERO_CAPACITY_FIELDS = {
    "min_provisioned_throughput",
    "max_provisioned_throughput",
    "provisioned_model_units",
    "min_provisioned_concurrency",
    "max_provisioned_concurrency",
}


def _ownership(value: Mapping[str, str]) -> dict[str, str]:
    """Require all target-specific tags before reading or creating resources."""
    if not isinstance(value, Mapping) or set(value) != _OWNER_KEYS:
        raise ValueError("ownership requires project, target, suffix and managed_by endpoint tags.")
    if any(
        not isinstance(item, str) or not item.strip() or len(item) > 256 for item in value.values()
    ):
        raise ValueError("ownership tag values must be nonempty strings of at most 256 characters.")
    if value["managed_by"] != "serving_endpoint":
        raise ValueError("managed_by ownership must be serving_endpoint.")
    return dict(value)


def _dictionary(value: Any) -> dict[str, Any]:
    """Use SDK serialization while preserving raw native telemetry responses."""
    return dict(value) if isinstance(value, Mapping) else value.as_dict()


def validate_serving_tags(tags: Mapping[str, str] | None) -> dict[str, str]:
    """Copy optional cost tags after checking lengths and reserved ownership collisions."""
    if tags is None:
        return {}
    if not isinstance(tags, Mapping):
        raise ValueError("Serving tags must be a mapping of nonempty string keys and values.")
    if set(tags) & _OWNER_KEYS:
        raise ValueError("Serving tags cannot override reserved ownership tag keys.")
    for key, value in tags.items():
        if not _valid_tag_text(key, 128) or not _valid_tag_text(value, 256):
            raise ValueError(
                "Serving tags require nonempty keys up to 128 and values up to 256 characters."
            )
    return dict(tags)


def _valid_tag_text(value: Any, limit: int) -> bool:
    """Reject empty or control-bearing billing metadata before contacting the service."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= limit
        and value.isprintable()
    )


def _tag_matches(endpoint: dict[str, Any], key: str, value: str) -> bool:
    """Read exactly one value for a requested tag while retaining unrelated metadata."""
    return [tag.get("value") for tag in endpoint.get("tags") or [] if tag.get("key") == key] == [
        value
    ]


def _require_owner(endpoint: dict[str, Any], ownership: dict[str, str]) -> None:
    """Reject foreign, missing or duplicate ownership tags without adopting names."""
    for key, value in ownership.items():
        if not _tag_matches(endpoint, key, value):
            raise ValueError("Serving endpoint is not owned by this project and target.")


def _require_route(config: dict[str, Any], entity_name: str) -> None:
    """Admit exactly one named served entity receiving all traffic."""
    routes = (config.get("traffic_config") or {}).get("routes") or []
    if len(routes) != 1:
        raise ValueError("Owned serving endpoint requires one 100 percent traffic route.")
    route = routes[0]
    names = [route[key] for key in ("served_entity_name", "served_model_name") if key in route]
    if set(names) != {entity_name}:
        raise ValueError("Owned serving endpoint traffic differs from its named entity.")
    if type(route.get("traffic_percentage")) is not int or route["traffic_percentage"] != 100:
        raise ValueError("Owned serving endpoint requires one 100 percent traffic route.")


def _require_owned_config(endpoint: dict[str, Any], plan: PinnedEndpointPlan) -> dict[str, Any]:
    """Reject pending work, foreign models and runtime settings before replacing config."""
    state = endpoint.get("state") or {}
    if endpoint.get("pending_config") or state.get("config_update") != "NOT_UPDATING":
        raise RuntimeError("Serving endpoint has a pending or unsettled config update.")
    config = endpoint.get("config") or {}
    entities = config.get("served_entities") or []
    if endpoint.get("name") != plan.spec.endpoint_name or len(entities) != 1:
        raise ValueError("Owned serving endpoint must have exactly one served entity.")
    entity = entities[0]
    if not entity.get("name") or entity.get("entity_name") != plan.spec.model_name:
        raise ValueError("Owned serving endpoint must name one version of the selected model.")
    _require_route(config, entity["name"])
    _require_runtime_settings(config, entity, plan)
    require_pinned_endpoint_logging(endpoint, plan)
    return entity


def _require_runtime_settings(config: dict, entity: dict, plan: PinnedEndpointPlan) -> None:
    """Avoid dropping undeclared runtime configuration during a managed version swap."""
    config_keys = {
        "served_entities",
        "served_models",
        "traffic_config",
        "config_version",
        "start_time",
        "auto_capture_config",
    }
    if any(config[key] not in (None, {}, []) for key in set(config) - config_keys):
        raise ValueError("Owned serving endpoint has unadmitted deployment settings.")
    expected = set(plan.config["config"]["served_entities"][0])
    metadata = {"name", "creation_timestamp", "creator", "state", "type", "batching_config"}
    if any(
        not _inactive_entity_setting(key, entity[key]) for key in set(entity) - expected - metadata
    ):
        raise ValueError("Owned serving endpoint has unadmitted runtime settings.")
    _require_native_entity(entity)
    _require_inactive_legacy_capture(config)
    _require_legacy_entities(config, entity)


def _inactive_entity_setting(key: str, value: Any) -> bool:
    """Admit empty SDK fields and explicitly zero provisioned capacity on CPU serving."""
    if value in (None, {}, []):
        return True
    return key in _ZERO_CAPACITY_FIELDS and type(value) is int and value == 0


def _require_inactive_legacy_capture(config: dict[str, Any]) -> None:
    """A disabled legacy logging readback is harmless and is never sent in updates."""
    capture = config.get("auto_capture_config")
    if capture in (None, {}):
        return
    if not isinstance(capture, dict) or capture.get("enabled") is not False:
        raise ValueError("Owned serving endpoint has conflicting legacy payload logging.")


def _require_native_entity(entity: dict[str, Any]) -> None:
    """Admit native platform metadata and explicitly disabled default batching only."""
    if entity.get("type", "UC_MODEL") != "UC_MODEL":
        raise ValueError("Owned serving endpoint requires a native UC model.")
    batching = entity.get("batching_config")
    if batching not in (None, {}, {"enabled": False}):
        raise ValueError("Owned serving endpoint has unadmitted batching settings.")


def _require_legacy_entities(config: dict, entity: dict) -> None:
    """Admit SDK legacy aliases only when they describe the same served model."""
    aliases = config.get("served_models") or []
    if aliases:
        normalized = [
            {
                {"model_name": "entity_name", "model_version": "entity_version"}.get(k, k): v
                for k, v in alias.items()
            }
            for alias in aliases
        ]
        if normalized != [entity]:
            raise ValueError("Owned serving endpoint has conflicting legacy served models.")


def _named_plan(plan: PinnedEndpointPlan, name: str, owner: dict[str, str]) -> PinnedEndpointPlan:
    """Copy inspected configuration and add atomic ownership plus explicit routing."""
    config = deepcopy(plan.config)
    config["config"]["served_entities"][0]["name"] = name
    config["config"]["traffic_config"] = {
        "routes": [{"served_model_name": name, "traffic_percentage": 100}]
    }
    config["tags"] = [{"key": key, "value": value} for key, value in owner.items()]
    return replace(plan, config=config)


def ensure_owned_endpoint(
    client: Any,
    plan: PinnedEndpointPlan,
    *,
    ownership: Mapping[str, str],
    tags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create or update one owned pinned model, then require its exact settled readback.

    Creation includes ownership tags atomically. Existing foreign names, pending
    updates, multiple entities, routing and logging conflicts fail before writes.
    The caller owns concurrency between separate deployments of the same target.
    """
    from databricks.sdk.errors import NotFound  # noqa: PLC0415
    from databricks.sdk.service import serving  # noqa: PLC0415

    owner = _ownership(ownership)
    extra_tags = validate_serving_tags(tags)
    tags_updated = False
    try:
        endpoint = _dictionary(read_pinned_endpoint(client, plan))
    except NotFound:
        wanted = _named_plan(plan, plan.spec.endpoint_name, owner | extra_tags)
        final = create_pinned_endpoint(client, wanted)
        action, previous = "created", None
    else:
        _require_owner(endpoint, owner)
        entity = _require_owned_config(endpoint, plan)
        prior_spec = replace(plan.spec, model_version=entity.get("entity_version"))
        previous = prior_spec.model_version
        wanted = _named_plan(plan, entity["name"], owner | extra_tags)
        desired = wanted.config["config"]["served_entities"][0]
        tags_updated = _reconcile_tags(client, endpoint, extra_tags)
        action = "updated" if tags_updated else "unchanged"
        if any(entity.get(key) != value for key, value in desired.items()):
            client.serving_endpoints.update_config(
                plan.spec.endpoint_name,
                served_entities=[serving.ServedEntityInput.from_dict(desired)],
                traffic_config=serving.TrafficConfig.from_dict(
                    wanted.config["config"]["traffic_config"]
                ),
            )
            action = "updated"
        final = wait_for_pinned_endpoint_ready(client, wanted)
    final = _dictionary(final)
    _require_owner(final, owner)
    _require_owned_config(final, wanted)
    if any(not _tag_matches(final, key, value) for key, value in extra_tags.items()):
        raise ValueError("Serving endpoint requested extra tags differ from their exact readback.")
    return {
        "action": action,
        "previous_version": previous,
        "endpoint_name": plan.spec.endpoint_name,
        "model_version": plan.spec.model_version,
        "tags_updated": tags_updated,
    }


def _reconcile_tags(client: Any, endpoint: dict[str, Any], tags: dict[str, str]) -> bool:
    """Add or update requested extra tags without deleting any other endpoint metadata."""
    from databricks.sdk.service import serving  # noqa: PLC0415

    changed = [
        serving.EndpointTag(key=key, value=value)
        for key, value in tags.items()
        if not _tag_matches(endpoint, key, value)
    ]
    if not changed:
        return False
    client.serving_endpoints.patch(endpoint["name"], add_tags=changed)
    return True


def _sql_metadata(plan: ServingSQLFunctionPlan, owner: dict[str, str]) -> dict[str, Any]:
    """Keep endpoint ownership stable while fingerprinting SQL types and wire encoding."""
    schema = json.dumps(plan.schema_contract, sort_keys=True, separators=(",", ":"))
    return {
        "ownership": owner,
        "endpoint_name": plan.endpoint.spec.endpoint_name,
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
    }


def _require_function_owner(comment: Any, expected: dict[str, Any]) -> dict[str, Any]:
    """Treat unrelated or malformed function comments as foreign ownership."""
    try:
        if not isinstance(comment, str) or not comment.startswith(_COMMENT_PREFIX):
            raise ValueError("missing managed comment")
        actual = json.loads(comment[len(_COMMENT_PREFIX) :])
        if any(actual.get(key) != expected[key] for key in ("ownership", "endpoint_name")):
            raise ValueError("different owner")
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("SQL function is not owned by this endpoint and target.") from exc
    return actual


def ensure_owned_sql_function(
    spark: Any, client: Any, plan: ServingSQLFunctionPlan, *, ownership: Mapping[str, str]
) -> dict[str, str]:
    """Enroll stable SQL once and replace only a changed schema for this owned endpoint."""
    from databricks.sdk.errors import NotFound  # noqa: PLC0415

    owner = _ownership(ownership)
    metadata = _sql_metadata(plan, owner)
    try:
        function = _dictionary(client.functions.get(plan.function_name))
    except NotFound:
        action = "created"
    else:
        actual = _require_function_owner(function.get("comment"), metadata)
        action = "unchanged" if actual == metadata else "updated"
    endpoint = _dictionary(require_pinned_endpoint_ready(client, plan.endpoint))
    _require_owner(endpoint, owner)
    _require_owned_config(endpoint, plan.endpoint)
    if action != "unchanged":
        comment = _COMMENT_PREFIX + json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        spark.sql(plan.render_create_sql(comment=comment, replace=action == "updated")).collect()
    return {"action": action, "function_name": plan.function_name}


def validate_serving_principals(principals: Any) -> list[dict[str, str]]:
    """Normalize principal dictionaries to one identity plus level CAN_QUERY before writes."""
    if not isinstance(principals, Sequence) or isinstance(principals, (str, bytes)):
        raise ValueError("permissions must be a list of principal dictionaries.")
    return [_principal(principal) for principal in principals]


def _principal(value: Any) -> dict[str, str]:
    """Validate one additive ACL entry and preserve its literal principal identity."""
    if not isinstance(value, Mapping) or set(value) - _PRINCIPAL_KEYS - {"level"}:
        raise ValueError("permissions require one supported principal and optional level.")
    keys = set(value) & _PRINCIPAL_KEYS
    if len(keys) != 1 or value.get("level", "CAN_QUERY") != "CAN_QUERY":
        raise ValueError("permissions require one principal and level CAN_QUERY.")
    key = next(iter(keys))
    name = value[key]
    if not isinstance(name, str) or not name.strip() or any(ord(char) < 32 for char in name):
        raise ValueError("permission principal must be a nonempty string without controls.")
    return {key: name, "level": "CAN_QUERY"}


def grant_serving_access(
    spark: Any,
    client: Any,
    plan: PinnedEndpointPlan,
    *,
    principals: Any,
    function_name: str | None = None,
) -> None:
    """Add endpoint CAN_QUERY and optional SQL EXECUTE without replacing or revoking ACLs."""
    from databricks.sdk.service import serving  # noqa: PLC0415

    entries = validate_serving_principals(principals)
    function = (
        build_serving_sql_function(plan, function_name) if function_name is not None else None
    )
    if not entries:
        return
    endpoint = _dictionary(client.serving_endpoints.get(plan.spec.endpoint_name))
    if not endpoint.get("id"):
        raise ValueError("Serving endpoint readback requires its ID for additive grants.")
    acl = [
        serving.ServingEndpointAccessControlRequest.from_dict(_sdk_permission(entry))
        for entry in entries
    ]
    client.serving_endpoints.update_permissions(endpoint["id"], access_control_list=acl)
    if function is not None:
        for entry in entries:
            name = next(value for key, value in entry.items() if key != "level")
            principal = "`" + name.replace("`", "``") + "`"
            spark.sql(f"GRANT EXECUTE ON FUNCTION {function.quoted_name} TO {principal}").collect()


def _sdk_permission(entry: dict[str, str]) -> dict[str, str]:
    """Translate the validated configuration level to the SDK permission field."""
    return {key: value for key, value in entry.items() if key != "level"} | {
        "permission_level": "CAN_QUERY"
    }
