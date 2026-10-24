"""Create or update a target-owned endpoint and prove SQL/REST prediction parity."""

import hashlib
import re
from dataclasses import fields
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ...mlflow.registration.registry import download_registered_package
from ...mlflow.shared._nullable_transport import encode_frame, transport_spec
from ..lifecycle.workflow import bind_target_name
from ..projects.yaml_config import read_yaml_mapping
from ..serving.contracts import PinnedEndpointPlan, PinnedEndpointSpec, is_uc_identifier
from ..serving.endpoints import prepare_pinned_endpoint, query_named_records, validate_named_rows
from ..serving.managed_endpoint import (
    ensure_owned_endpoint,
    ensure_owned_sql_function,
    grant_serving_access,
    validate_serving_principals,
    validate_serving_tags,
)
from ..serving.sql_functions import ServingSQLFunctionPlan, build_serving_sql_function
from .serving_enrollment import (
    build_serving_monitoring_configs,
    enroll_verified_serving,
    serving_monitoring_namespace,
    validate_serving_label_sources,
)

_SPEC_FIELDS = {field.name for field in fields(PinnedEndpointSpec)} - {"model_version"}
_SETTINGS_FIELDS = _SPEC_FIELDS | {
    "enabled",
    "follow_champion",
    "tracking_uri",
    "registry_uri",
    "sql_function",
    "permissions",
    "function_permissions",
    "verification",
}
_BINDINGS = ("catalog", "input_schema", "output_schema", "metadata_schema", "resource_suffix")


def _known_fields(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    """Reject misspellings instead of silently choosing a different deployment behavior."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    if set(value) - allowed:
        raise ValueError(f"Unknown {label} fields: {sorted(set(value) - allowed)}.")
    return value.copy()


def _boolean(value: Any, label: str) -> bool:
    """Require an explicit boolean rather than accepting truthy strings."""
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean.")
    return value


def _substitute(value: str, bindings: dict[str, str]) -> str:
    """Resolve target placeholders without evaluating format expressions."""
    if not isinstance(value, str):
        raise ValueError("Serving names must be strings.")
    for key, replacement in bindings.items():
        value = value.replace("{" + key + "}", replacement)
    if "{" in value or "}" in value:
        raise ValueError("Unresolved serving name placeholder.")
    return value


def _bound_settings(settings: dict, values: dict[str, str]) -> dict:
    """Apply UC target ownership and an endpoint-safe spelling of the same suffix."""
    bindings = {key: values.get(key, "") for key in _BINDINGS}
    if any(not is_uc_identifier(bindings[key]) for key in _BINDINGS[:-1]):
        raise ValueError("Serving requires resolved catalog and schema target bindings.")
    suffix = bindings["resource_suffix"]
    if suffix and (not suffix.startswith("_") or not is_uc_identifier(suffix)):
        raise ValueError("Serving resource_suffix must be an underscore-prefixed identifier.")
    settings["model_name"] = bind_target_name(
        "model_name", settings["model_name"], bindings, suffix
    )
    settings["endpoint_name"] = _bound_endpoint_name(
        settings["endpoint_name"],
        {
            **bindings,
            "resource_suffix": suffix.replace("_", "-"),
            "bundle_target": values.get("bundle_target", "").replace("_", "-"),
        },
    )
    for key in ("logging_catalog", "logging_schema", "logging_table_prefix"):
        settings[key] = _substitute(settings.get(key, ""), bindings)
    sql = _known_fields(settings.get("sql_function", {}), {"enabled", "name"}, "sql_function")
    if _boolean(sql.get("enabled", True), "sql_function.enabled"):
        sql["name"] = bind_target_name("model_name", sql["name"], bindings, suffix)
    settings["sql_function"] = {"enabled": True, **sql}
    settings["ownership"] = _owner(values, suffix)
    settings["tags"] = _cost_tags(values)
    return settings


def _bound_endpoint_name(template: str, bindings: dict[str, str]) -> str:
    """Bound generated personal names while preserving explicit operator-selected names."""
    name = _substitute(template, bindings)
    generated = template.endswith("-{bundle_target}-endpoint{resource_suffix}")
    if len(name) > 63 and generated and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        return name[:46] + "-" + hashlib.sha256(name.encode()).hexdigest()[:16]
    return name


def _cost_tags(values: dict[str, str]) -> dict[str, str]:
    """Require a real allocation value instead of publishing the wizard placeholder."""
    value = values.get("cost_tag_value", "")
    if value == "REPLACE_WITH_COST_TAG" or "${" in value:
        raise ValueError("Set cost_tag_value in the target Bundle variables before serving.")
    return validate_serving_tags({values.get("cost_tag_key", ""): value})


def _owner(values: dict[str, str], suffix: str) -> dict[str, str]:
    """Reserve four plain tags for exact project and target ownership."""
    project, target = values.get("bundle_project", ""), values.get("bundle_target", "")
    if not project or not target or any("${" in value for value in (project, target)):
        raise ValueError("Serving requires resolved bundle_project and bundle_target.")
    return {
        "project": project,
        "target": target,
        "suffix": suffix or "shared",
        "managed_by": "serving_endpoint",
    }


def target_permissions(permissions: Any, target: str) -> list[dict[str, str]]:
    """Return this target's grants; a list applies to every target.

    A mapping keyed by Bundle target lets each environment grant its own groups.
    Every target's entries are validated, so a malformed prod entry fails in test
    too. An unlisted target, such as a personal development target, grants nothing.
    """
    if not isinstance(permissions, dict):
        return validate_serving_principals(permissions)
    selected: list[dict[str, str]] = []
    for name, entries in permissions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("permissions keys must be Bundle target names.")
        if not isinstance(entries, list):
            raise ValueError(f"permissions.{name} must be a list of principal dictionaries.")
        validated = validate_serving_principals(entries)
        if name == target:
            selected = validated
    return selected


def _validate_settings(settings: dict[str, Any]) -> None:
    """Check behavioral controls and compute before downloading a model or mutating resources."""
    _boolean(settings.get("enabled"), "serving_endpoint.enabled")
    _boolean(settings.get("follow_champion", True), "serving_endpoint.follow_champion")
    verification = _known_fields(
        settings.get("verification", {"sample_rows": 20}), {"sample_rows"}, "verification"
    )
    if (
        type(verification.get("sample_rows")) is not int
        or not 1 <= verification["sample_rows"] <= 1000
    ):
        raise ValueError("verification.sample_rows must be an integer from 1 to 1000.")
    settings["verification"] = verification
    target = settings["ownership"]["target"]
    settings["permissions"] = target_permissions(settings.get("permissions", []), target)
    # Omitted: the endpoint callers also get the SQL function's EXECUTE grant.
    settings["function_permissions"] = (
        target_permissions(settings["function_permissions"], target)
        if "function_permissions" in settings
        else settings["permissions"]
    )
    settings["tags"] = validate_serving_tags(settings.get("tags"))
    _spec(settings, "1")


def _spec(settings: dict[str, Any], version: str) -> PinnedEndpointSpec:
    """Reuse the established concrete selector, logging and CPU admission contract."""
    return PinnedEndpointSpec(
        model_version=version, **{key: settings[key] for key in _SPEC_FIELDS if key in settings}
    )


def _refuse_shared_resources(settings: dict, document: dict) -> None:
    """Do not compete with rollout/online setup or mix their payload sinks."""
    online = document.get("online_feature_lookup", {}).get("endpoint", {})
    rollout = document.get("rollout", {}).get("monitoring", {})
    if settings["endpoint_name"] in {online.get("endpoint_name"), rollout.get("serving_endpoint")}:
        raise ValueError("Endpoint is already managed by rollout or online_feature_lookup setup.")
    if settings.get("logging_mode") == "disabled":
        return
    keys = ("logging_catalog", "logging_schema", "logging_table_prefix")
    if online.get("logging_mode", "telemetry") != "disabled" and all(
        settings[key] == online.get(key) for key in keys
    ):
        raise ValueError("Serving endpoints require separate logging tables.")
    if _spec(settings, "1").inference_table in {
        rollout.get("source_table"),
        rollout.get("prediction_table"),
    }:
        raise ValueError("Serving endpoint logging table is already used by rollout.")


def read_serving_endpoint_settings(values: dict[str, str]) -> dict[str, Any]:
    """Load the optional deployed YAML and bind names before any remote side effects."""
    document = read_yaml_mapping(values["serving_config_path"])
    settings = _known_fields(document.get("serving_endpoint"), _SETTINGS_FIELDS, "serving_endpoint")
    if not _boolean(settings.get("enabled"), "serving_endpoint.enabled"):
        return settings
    settings = _bound_settings(settings, values)
    _validate_settings(settings)
    _refuse_shared_resources(settings, document)
    return settings


def sample_serving_rows(
    spark: Any,
    source_table: str,
    schema: tuple,
    *,
    rows: int,
    record_key_columns: tuple[str, ...] = (),
) -> list[dict]:
    """Collect bounded raw inputs without rounding identities or nullable integers."""
    from skyulf.engines.spark_engine import normalize_spark_decimals  # noqa: PLC0415

    if len(source_table.split(".")) != 3 or not all(
        is_uc_identifier(part) for part in source_table.split(".")
    ):
        raise ValueError("Scoring source must be a concrete catalog.schema.table.")
    columns = [f"`{name.replace('`', '``')}`" for name, _ in schema]
    sort_name = "_serving_sample_" + uuid4().hex
    selected = spark.table(source_table).selectExpr(
        *columns, f"xxhash64({', '.join(columns)}) AS `{sort_name}`"
    )
    selected = selected.orderBy(sort_name, *columns).limit(rows).selectExpr(*columns)
    selected, _ = normalize_spark_decimals(
        selected, columns=(name for name, _ in schema if name not in record_key_columns)
    )
    sample = selected.collect()
    if not sample:
        raise ValueError("Serving smoke sample is empty; no endpoint was created or updated.")
    return [row.asDict(recursive=True) for row in sample]


def _smoke_frame(raw: list[dict], plan: PinnedEndpointPlan) -> pd.DataFrame:
    """Restore saved dtypes without turning missing boolean inputs into real values."""
    columns = {}
    for name, dtype in plan.input_schema:
        values = [row[name] for row in raw]
        if dtype == "bool" and any(pd.isna(value) for value in values):
            raise ValueError(
                f"Serving sample contains null in nonnullable boolean column {name!r}."
            )
        columns[name] = pd.Series(values, dtype=dtype)
    return pd.DataFrame(columns)


def _smoke_inputs(raw: list[dict], plan: PinnedEndpointPlan) -> tuple[pd.DataFrame, list[dict]]:
    """Use saved native dtypes before the existing lossless nullable transport encoder."""
    frame = _smoke_frame(raw, plan)
    encoded = encode_frame(frame, transport_spec(plan.input_schema))
    records = [
        {name: row[name] for name in plan.input_columns}
        for row in encoded.astype(object).where(pd.notna(encoded), None).to_dict("records")
    ]
    validate_named_rows(records, plan)
    return encoded, records


def verify_serving_predictions(expected: pd.DataFrame, actual: Any, *, channel: str) -> int:
    """Compare every row/field; only floating transport allows tiny serialization noise.

    Floating outputs use rtol=1e-10 and atol=1e-12 for JSON/SQL binary conversion.
    Integer, boolean, label and status values remain exact; nulls must match.
    """
    if not isinstance(expected, pd.DataFrame) or expected.empty:
        raise ValueError(f"{channel} smoke requires nonempty named reference predictions.")
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or any(not isinstance(row, dict) or set(row) != set(expected.columns) for row in actual)
    ):
        raise ValueError(f"{channel} prediction row count or output columns differ.")
    observed = pd.DataFrame(actual, columns=expected.columns, dtype=object)
    try:
        for name in expected.columns:
            _compare_column(expected[name].reset_index(drop=True), observed[name])
    except (AssertionError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{channel} predictions differ from the concrete registered version."
        ) from exc
    return len(expected)


def _compare_column(expected: pd.Series, actual: pd.Series) -> None:
    """Avoid floating-point tolerance on integer identities and classification labels."""
    _require_scalar_kind(expected, actual)
    if pd.api.types.is_float_dtype(expected.dtype):
        observed = actual.to_numpy(dtype=float, na_value=np.nan)
        reference = expected.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(observed).any() or np.isinf(reference).any():
            raise ValueError("Infinite predictions cannot be verified for JSON/SQL serving.")
        np.testing.assert_allclose(
            observed,
            reference,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
    else:
        expected = expected.astype(object).where(expected.notna(), None)
        actual = actual.astype(object).where(actual.notna(), None)
        pd.testing.assert_series_equal(
            expected, actual, check_dtype=False, check_names=False, check_exact=True
        )


def _require_scalar_kind(expected: pd.Series, actual: pd.Series) -> None:
    """Do not let Python equality or numeric coercion conceal a changed response type."""
    if pd.api.types.is_bool_dtype(expected.dtype):
        allowed = (bool, np.bool_)
    elif pd.api.types.is_integer_dtype(expected.dtype):
        allowed = (int, np.integer)
    elif pd.api.types.is_float_dtype(expected.dtype):
        allowed = (int, float, np.integer, np.floating)
    else:
        return
    for value in actual:
        if pd.isna(value):
            continue
        wrong_bool = isinstance(value, (bool, np.bool_)) and bool not in allowed
        if not isinstance(value, allowed) or wrong_bool:
            raise ValueError("Response scalar type differs from the saved output.")


def _load_pyfunc(settings: dict, spec: PinnedEndpointSpec, registry: Any) -> Any:
    """Load the same concrete package through explicit clients, without global URI changes."""
    import mlflow  # noqa: PLC0415 - optional integration dependency

    package = download_registered_package(
        mlflow,
        registry,
        spec.model_name,
        spec.model_version,
        settings.get("tracking_uri", "databricks"),
    )
    return mlflow.pyfunc.load_model(package)


def _sql_input_frame(
    spark: Any, function: ServingSQLFunctionPlan, raw: list[dict]
) -> tuple[Any, str]:
    """Validate native SQL values before changing endpoint or function resources."""
    ordinal = "_serving_row_" + uuid4().hex
    schema = ", ".join(
        [
            f"{ordinal} BIGINT",
            *(f"{name} {dtype}" for name, dtype in function.schema_contract["inputs"]),
        ]
    )
    native = _smoke_frame(raw, function.endpoint)
    rows = native.astype(object).where(pd.notna(native), None).to_dict("records")
    data = [
        (index, *(row[name] for name in function.endpoint.input_columns))
        for index, row in enumerate(rows)
    ]
    frame = spark.createDataFrame(data, schema=schema)
    return frame, ordinal


def _sql_predictions(function: ServingSQLFunctionPlan, sample: tuple[Any, str]) -> list[dict]:
    """Call the UC function on the validated in-memory sample, retaining row order."""
    frame, ordinal = sample
    result = (
        frame.selectExpr(ordinal, f"{function.call_sql()} AS response").orderBy(ordinal).collect()
    )
    return [row["response"].asDict(recursive=True) for row in result]


def _verify_sql_predictions(
    function: ServingSQLFunctionPlan | None,
    sample: tuple[Any, str] | None,
    expected: pd.DataFrame,
) -> int:
    """Verify the enabled SQL path using only its prevalidated input sample."""
    if function is None:
        return 0
    if sample is None:
        raise ValueError("SQL verification requires its prepared input sample.")
    return verify_serving_predictions(expected, _sql_predictions(function, sample), channel="SQL")


def _request_examples(
    plan: Any, function: ServingSQLFunctionPlan | None, source: str, record: dict
) -> dict:
    """Show matching SQL/REST/SDK shapes using one sample record and no credentials."""
    return {
        "sql": None
        if function is None
        else f"SELECT {function.call_sql()} AS prediction FROM {source};",
        "rest_path": f"/serving-endpoints/{plan.spec.endpoint_name}/invocations",
        "rest_body": {"dataframe_records": [record]},
        "python": "from databricks.sdk import WorkspaceClient\nclient = WorkspaceClient()\nrecords = "
        + repr([record])
        + "\nresult = client.serving_endpoints.query(\n    name="
        + repr(plan.spec.endpoint_name)
        + ",\n    dataframe_records=records,  # Same named rows as rest_body\n)\nprint(result.predictions)",
    }


def run_serving_endpoint(
    spark: Any,
    settings: dict,
    *,
    source_table: str,
    client: Any,
    registry_client: Any,
    model_version: str = "",
    monitoring_values: dict[str, str] | None = None,
) -> dict:
    """Prepare and validate the whole request before updating owned resources, then prove parity."""
    _known_fields(settings, _SETTINGS_FIELDS | {"ownership", "tags"}, "serving_endpoint")
    if not settings["enabled"]:
        return {"status": "DISABLED"}
    _validate_settings(settings)
    monitoring_values = monitoring_values or {}
    namespace = serving_monitoring_namespace(
        settings.get("logging_mode", "telemetry"), monitoring_values
    )
    version = model_version or str(
        registry_client.get_model_version_by_alias(settings["model_name"], "champion").version
    )
    spec = _spec(settings, version)
    plan = prepare_pinned_endpoint(
        spec,
        tracking_uri=settings.get("tracking_uri", "databricks"),
        registry_uri=settings.get("registry_uri", "databricks-uc"),
    )
    sql = settings["sql_function"]
    function = build_serving_sql_function(plan, sql["name"]) if sql["enabled"] else None
    raw = sample_serving_rows(
        spark,
        source_table,
        plan.input_schema,
        rows=settings["verification"]["sample_rows"],
        record_key_columns=plan.record_key_columns,
    )
    frame, records = _smoke_inputs(raw, plan)
    sql_sample = _sql_input_frame(spark, function, raw) if function is not None else None
    model = _load_pyfunc(settings, spec, registry_client)
    expected = model.predict(frame)
    # Validate the reference too, before a mutation can leave an untestable endpoint.
    verify_serving_predictions(expected, expected.to_dict("records"), channel="Reference")
    configs = build_serving_monitoring_configs(
        spec,
        settings,
        namespace,
        monitoring_values,
        {} if namespace is None else model.metadata.metadata,
    )
    validate_serving_label_sources(spark, configs)
    owner = settings["ownership"]
    result = ensure_owned_endpoint(client, plan, ownership=owner, tags=settings["tags"])
    sql_result = (
        ensure_owned_sql_function(spark, client, function, ownership=owner) if function else None
    )
    grant_serving_access(
        spark,
        client,
        plan,
        principals=settings["permissions"],
        function_principals=settings["function_permissions"],
        function_name=None if function is None else function.function_name,
    )
    rest = query_named_records(client, plan, records)
    counts = {
        "sampled": len(raw),
        "rest_compared": verify_serving_predictions(expected, rest.predictions, channel="REST"),
        "sql_compared": _verify_sql_predictions(function, sql_sample, expected),
    }
    return {
        **result,
        "status": "READY",
        "model_uri": spec.model_uri,
        "logging_mode": spec.logging_mode,
        "payload_table": None if spec.logging_mode == "disabled" else spec.inference_table,
        "sql_function": sql_result,
        "monitoring": enroll_verified_serving(
            spark,
            namespace,
            configs,
            monitoring_values,
            tracking_uri=settings.get("tracking_uri", "databricks"),
            registry_uri=settings.get("registry_uri", "databricks-uc"),
        ),
        "verification": counts,
        "examples": _request_examples(plan, function, source_table, records[0]),
    }


def run_serving_endpoint_notebook(spark: Any, dbutils: Any) -> dict[str, Any]:
    """Run with the job identity and publish a small machine-readable result."""
    values = dbutils.widgets.getAll()
    settings = read_serving_endpoint_settings(values)
    if not settings["enabled"]:
        return {"status": "DISABLED"}
    _validate_job_timeout(values, settings)
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    inference = read_yaml_mapping(values["inference_config_path"])
    bindings = {key: values[key] for key in _BINDINGS}
    source = bind_target_name(
        "score_source_table", inference["score_source_table"], bindings, bindings["resource_suffix"]
    )
    registry = MlflowClient(
        tracking_uri=settings.get("tracking_uri", "databricks"),
        registry_uri=settings.get("registry_uri", "databricks-uc"),
    )
    result = run_serving_endpoint(
        spark,
        settings,
        source_table=source,
        client=WorkspaceClient(),
        registry_client=registry,
        model_version=values.get("model_version", ""),
        monitoring_values=values,
    )
    dbutils.jobs.taskValues.set(key="serving_result", value=result)
    return result


def _validate_job_timeout(values: dict[str, str], settings: dict) -> None:
    """Reserve a bounded inspection/smoke allowance beyond the endpoint readiness deadline."""
    raw = values.get("job_timeout_s", "")
    if not raw.isdecimal() or int(raw) < settings.get("ready_timeout_s", 3600) + 300:
        raise ValueError(
            "serving_timeout_seconds must be at least ready_timeout_s + 300; update the Bundle variable and redeploy."
        )
