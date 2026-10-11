"""Compose two admitted concrete model versions without weakening pinned serving."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..shared.json_contracts import finite_json_digest
from .contracts import PinnedEndpointPlan
from .endpoints import endpoint_ready


@dataclass(frozen=True, slots=True)
class RolloutEndpointPlan:
    """Retain both inspected packages and the explicit initial endpoint request."""

    champion: PinnedEndpointPlan
    challenger: PinnedEndpointPlan
    config: dict[str, Any]


def rollout_traffic(challenger_percentage: int) -> dict[str, Any]:
    """Allocate integer percentage points to exactly the two named served entities."""
    if type(challenger_percentage) is not int or not 0 <= challenger_percentage <= 100:
        raise ValueError("challenger_percentage must be an integer from 0 to 100.")
    return {
        "routes": [
            {"served_model_name": "champion", "traffic_percentage": 100 - challenger_percentage},
            {"served_model_name": "challenger", "traffic_percentage": challenger_percentage},
        ]
    }


def _validate_plans(champion: PinnedEndpointPlan, challenger: PinnedEndpointPlan) -> None:
    """Require a shared transport contract and distinct immutable releases."""
    if not isinstance(champion, PinnedEndpointPlan) or not isinstance(
        challenger, PinnedEndpointPlan
    ):
        raise TypeError("Rollout requires two admitted PinnedEndpointPlan values.")
    if champion.spec.endpoint_name != challenger.spec.endpoint_name:
        raise ValueError("Rollout plans must select the same endpoint.")
    if champion.spec.model_uri == challenger.spec.model_uri:
        raise ValueError("Rollout requires distinct concrete model versions.")
    if not champion.input_schema or not champion.output_schema:
        raise ValueError("Rollout requires complete named input/output schemas.")
    _matching_contract(champion, challenger)


def _matching_contract(champion: PinnedEndpointPlan, challenger: PinnedEndpointPlan) -> None:
    """Bind both prediction meanings and evidence destinations across the releases."""
    for field in ("input_columns", "input_schema", "output_schema", "online_contract"):
        if getattr(champion, field) != getattr(challenger, field):
            raise ValueError(f"Rollout package {field} differ.")
    first = {key: value for key, value in champion.config.items() if key != "config"}
    second = {key: value for key, value in challenger.config.items() if key != "config"}
    if first != second or champion.spec.logging_mode != challenger.spec.logging_mode:
        raise ValueError("Rollout plans must use the same endpoint logging configuration.")


def _served_entity(plan: PinnedEndpointPlan, name: str) -> dict[str, Any]:
    """Name an already admitted single served package without changing its compute."""
    if plan.config.get("name") != plan.spec.endpoint_name:
        raise ValueError("Rollout configuration differs from its endpoint selector.")
    logging = "telemetry_config" if plan.spec.logging_mode == "telemetry" else "ai_gateway"
    if set(plan.config) != {"name", "config", logging}:
        raise ValueError("Rollout configuration contains unadmitted endpoint settings.")
    config = plan.config.get("config", {})
    entities = config.get("served_entities", [])
    if len(entities) != 1 or set(config) != {"served_entities"}:
        raise ValueError("Rollout composition requires an unmodified pinned endpoint plan.")
    entity = entities[0]
    allowed = {
        "entity_name",
        "entity_version",
        "workload_type",
        "workload_size",
        "scale_to_zero_enabled",
    }
    if set(entity) != allowed:
        raise ValueError("Rollout configuration contains unadmitted served entity settings.")
    if (entity.get("entity_name"), entity.get("entity_version")) != (
        plan.spec.model_name,
        plan.spec.model_version,
    ):
        raise ValueError("Rollout package configuration differs from its concrete selector.")
    return deepcopy(entity) | {"name": name}


def build_rollout_endpoint(
    champion: PinnedEndpointPlan, challenger: PinnedEndpointPlan
) -> RolloutEndpointPlan:
    """Build an isolated 100/0 A/B endpoint from two independently inspected plans.

    Prepare both inputs through the existing pinned-package admission using the
    new rollout endpoint's name. This does not adopt, update or repurpose an
    existing pinned REST/SQL endpoint. Model-set weights remain inside each
    package; endpoint traffic chooses the entire served package.
    """
    _validate_plans(champion, challenger)
    config = deepcopy(champion.config)
    config["config"] = {
        "served_entities": [
            _served_entity(champion, "champion"),
            _served_entity(challenger, "challenger"),
        ],
        "traffic_config": rollout_traffic(0),
    }
    return RolloutEndpointPlan(deepcopy(champion), deepcopy(challenger), config)


def _endpoint_dict(endpoint: Any) -> dict[str, Any]:
    """Accept the authenticated raw response or the SDK's documented serializer."""
    if isinstance(endpoint, dict):
        return deepcopy(endpoint)
    if callable(getattr(endpoint, "as_dict", None)):
        return endpoint.as_dict()
    raise TypeError("Endpoint readback must be a dictionary or an SDK response.")


def _routes_match(config: dict[str, Any], wanted: dict[str, Any]) -> None:
    """Reject unknown, repeated or missing routes, including zero-share entities."""
    actual = config.get("traffic_config", {}).get("routes")
    expected = wanted["routes"]
    if not isinstance(actual, list) or len(actual) != 2:
        raise ValueError("Rollout traffic routes differ from the expected allocation.")
    actual = [_normalized_route(route) for route in actual]
    if any(route not in expected for route in actual) or actual[0] == actual[1]:
        raise ValueError("Rollout traffic routes differ from the expected allocation.")
    if any(type(route.get("traffic_percentage")) is not int for route in actual):
        raise ValueError("Rollout traffic percentages must be integers.")


def _normalized_route(route: Any) -> dict[str, Any]:
    """Accept the native entity-name alias only when it agrees with the model name."""
    required = {"served_model_name", "traffic_percentage"}
    if not isinstance(route, dict) or set(route) not in (
        required,
        required | {"served_entity_name"},
    ):
        raise ValueError("Rollout traffic route contains missing or unadmitted fields.")
    if route.get("served_entity_name", route["served_model_name"]) != route["served_model_name"]:
        raise ValueError("Rollout traffic route name aliases differ.")
    return {key: route[key] for key in required}


def _require_entity_settings(entity: dict[str, Any], expected: dict[str, Any]) -> None:
    """Reject undeclared runtime settings while allowing native deployment metadata."""
    _require_native_defaults(entity)
    metadata = {"creation_timestamp", "creator", "state", "type", "batching_config"}
    unexpected = set(entity) - set(expected) - metadata
    if any(entity[key] not in (None, {}, []) for key in unexpected):
        raise ValueError("Rollout configuration contains unadmitted served entity settings.")
    if any(entity.get(key) != value for key, value in expected.items()):
        raise ValueError("Rollout served entity configuration differs from its plan.")


def _require_native_defaults(entity: dict[str, Any]) -> None:
    """Accept only native UC model metadata and explicitly disabled default batching."""
    if entity.get("type", "UC_MODEL") != "UC_MODEL":
        raise ValueError("Rollout configuration requires native UC_MODEL entities.")
    batching = entity.get("batching_config")
    if batching is None or batching == {}:
        return
    if (
        not isinstance(batching, dict)
        or set(batching) != {"enabled"}
        or batching["enabled"] is not False
    ):
        raise ValueError("Rollout configuration contains unadmitted batching settings.")


def _require_config_shape(config: dict[str, Any]) -> None:
    """Allow native configuration metadata without adopting another serving surface."""
    _require_legacy_aliases(config)
    if "start_time" in config:
        stamp = config["start_time"]
        # Unix milliseconds through the final millisecond of UTC year 9999.
        if type(stamp) is not int or not 0 <= stamp <= 253402300799999:
            raise ValueError("Rollout configuration start_time must be bounded epoch milliseconds.")
    allowed = {"served_entities", "served_models", "traffic_config", "config_version", "start_time"}
    if any(config[key] not in (None, {}, []) for key in set(config) - allowed):
        raise ValueError("Rollout configuration contains unadmitted deployment settings.")


def _legacy_entity(entity: Any) -> dict[str, Any]:
    """Normalize only deprecated identity names while preserving every native field."""
    if not isinstance(entity, dict):
        raise ValueError("Rollout legacy configuration requires entity dictionaries.")
    names = {"model_name": "entity_name", "model_version": "entity_version"}
    normalized = {names.get(key, key): value for key, value in entity.items()}
    if len(normalized) != len(entity):
        raise ValueError("Rollout legacy configuration contains duplicate identity fields.")
    return normalized


def _require_legacy_aliases(config: dict[str, Any]) -> None:
    """Allow the deprecated view only as an exact equivalent of both native entities."""
    aliases = config.get("served_models")
    if aliases is None or aliases == []:
        return
    if not isinstance(aliases, list):
        raise ValueError("Rollout legacy configuration requires an entity list.")
    normalized = [finite_json_digest(_legacy_entity(entity)) for entity in aliases]
    entities = [finite_json_digest(entity) for entity in config.get("served_entities", [])]
    if len(normalized) != len(entities) or any(
        normalized.count(entity) != 1 for entity in entities
    ):
        raise ValueError("Rollout legacy configuration differs from its native entities.")


def rollout_endpoint_ready(
    endpoint: Any, plan: RolloutEndpointPlan, *, challenger_percentage: int
) -> bool:
    """Check both deployment state fields, exact routes, releases and telemetry.

    This readback is necessary after every mutation. The controller also binds
    the endpoint ID and config revision under shared writer admission so an
    out-of-band writer cannot be silently adopted.
    """
    wanted = rollout_traffic(challenger_percentage)
    value = _endpoint_dict(endpoint)
    state = value.get("state", {})
    update = state.get("config_update")
    if update in {"UPDATE_FAILED", "UPDATE_CANCELED"}:
        raise ValueError(f"Serving endpoint config update failed: {update}.")
    if state.get("ready") != "READY" or update != "NOT_UPDATING":
        return False
    config = value.get("config", {})
    _require_config_shape(config)
    entities = config.get("served_entities", [])
    if len(entities) != 2 or {entity.get("name") for entity in entities} != {
        "champion",
        "challenger",
    }:
        raise ValueError("Rollout requires exactly its champion and challenger entities.")
    _routes_match(config, wanted)
    for name, pinned in (("champion", plan.champion), ("challenger", plan.challenger)):
        projected = deepcopy(value)
        entity = next(entity for entity in entities if entity["name"] == name)
        _require_entity_settings(entity, _served_entity(pinned, name))
        projected["config"]["served_entities"] = [entity]
        endpoint_ready(projected, pinned)
    return True
