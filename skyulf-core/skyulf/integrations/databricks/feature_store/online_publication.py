"""Submit explicit latest-feature syncs through native Databricks Feature Engineering."""

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..data.admission import PublishAdmission
from ..shared._contracts import table_name
from .config import FeatureLookupSpec, FeatureTrainingSpec, validate_uc_name
from .lifecycle_config import deserialize_feature_spec, parse_feature_binding
from .runtime import feature_engineering_client
from .snapshots import snapshot_table
from .table_checks import validate_feature_tables


@dataclass(frozen=True, slots=True)
class OnlinePublicationSpec:
    """Select an existing store and one non-temporal latest-value Delta source.

    Provision the store and its permissions explicitly with the native SDK.
    Every table column is published; native model lookup may resolve the oldest
    online copy when a source has multiple publications. Use one publication per
    source for predictable routing. ``source_table_id`` detects table replacement.
    """

    source_table: str
    online_table: str
    online_store: str
    source_table_id: str

    def __post_init__(self) -> None:
        """Reject ambiguous resources before opening a Spark or workspace connection."""
        validate_uc_name(self.source_table)
        validate_uc_name(self.online_table)
        if self.source_table == self.online_table:
            raise ValueError("Online and source table names must differ.")
        if any(len(part.encode()) > 63 for part in self.online_table.split(".")):
            raise ValueError("Online table name components must be at most 63 bytes.")
        if not isinstance(self.online_store, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]{0,62}", self.online_store
        ):
            raise ValueError("online_store must be a safe 1-63 character store name.")
        if not isinstance(self.source_table_id, str) or not self.source_table_id.strip():
            raise ValueError("source_table_id must identify the existing Delta table.")


def _require_cdf_source(spark: Any, spec: OnlinePublicationSpec) -> None:
    """Bind the Delta table identity and incremental publication prerequisites."""
    detail = spark.sql(f"DESCRIBE DETAIL {table_name(spec.source_table)}").first()
    if detail is None or (detail["format"], detail["id"]) != ("delta", spec.source_table_id):
        raise ValueError("Online feature source must be the selected existing Delta table.")
    if detail["properties"].get("delta.enableChangeDataFeed") != "true":
        raise ValueError("TRIGGERED online publication requires Change Data Feed.")


def _source_contract(spark: Any, spec: OnlinePublicationSpec, client: Any) -> dict[str, Any]:
    """Validate CDF, schema and full primary keys using the existing feature checks."""
    _require_cdf_source(spark, spec)
    metadata = client.get_table(name=spec.source_table)
    if metadata.timestamp_keys:
        raise ValueError("Online publication requires a non-temporal latest-value source.")
    source = spark.table(spec.source_table)
    keys = tuple(metadata.primary_keys)
    if not keys or any(source.schema[name].nullable for name in keys):
        raise ValueError("Online feature primary keys must be declared NOT NULL.")
    features = tuple(name for name in source.columns if name not in keys)
    training = FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name=spec.source_table, lookup_key=keys, feature_names=features
            ),
        ),
        label=None,
    )
    snapshot = snapshot_table(spark, spec.source_table)
    if snapshot["table_id"] != spec.source_table_id:
        raise ValueError("Online feature source identity changed before validation.")
    validate_feature_tables(spark, source, training, client, [snapshot])
    return snapshot


def _available_store(client: Any, name: str) -> Any:
    """Require the explicitly provisioned store to be available before publication."""
    store = client.get_online_store(name=name)
    if store is None or store.name != name:
        raise ValueError("Online store does not exist; provision the named store explicitly.")
    state = getattr(store.state, "value", store.state)
    if state != "AVAILABLE":
        raise ValueError(f"Online store is not AVAILABLE: {state}.")
    return store


def validate_online_publications(
    spark: Any,
    binding: dict[str, Any],
    publications: tuple[OnlinePublicationSpec, ...],
    *,
    client: Any,
    feature_client: Any = None,
) -> None:
    """Verify existing accessible online copies against the model's saved lookup binding.

    Public UC metadata enumerates copies; native synced-table reads verify their
    effective stores and first completed sync. Enumeration is limited to the
    caller's visibility. Operators must grant full publication visibility and
    prevent concurrent publication changes during setup. No resources are created
    or published here, and a completed sync does not attest entity freshness.
    """
    binding = parse_feature_binding(binding)
    declared = {spec.source_table: spec for spec in publications}
    expected = {row["table_name"]: row for row in binding["lookup_evidence"]["feature_tables"]}
    if len(declared) != len(publications) or set(declared) != set(expected):
        raise ValueError("Publication bindings must cover each saved lookup source exactly once.")
    lookups = deserialize_feature_spec(binding["lookup_spec"])
    copies = _accessible_publications(client, set(expected))
    features = feature_engineering_client(feature_client)
    for name, spec in declared.items():
        if spec.source_table_id != expected[name]["table_id"]:
            raise ValueError("Publication Delta identity differs from the saved training binding.")
        if copies[name.casefold()] != [spec.online_table]:
            raise ValueError("Publication must be the sole accessible online copy of its source.")
        _available_store(features, spec.online_store)
        _source_contract(spark, spec, features)
        primary_keys = tuple(features.get_table(name=name).primary_keys)
        _validate_existing_publication(client, spec, lookups, primary_keys)


def _accessible_publications(client: Any, sources: set[str]) -> dict[str, list[str]]:
    """Enumerate public UC source properties without relying on unavailable list APIs."""
    found: dict[str, list[str]] = {name.casefold(): [] for name in sources}
    for catalog in client.catalogs.list(include_browse=True):
        kind = getattr(catalog, "catalog_type", None)
        if getattr(kind, "value", kind) == "SYSTEM_CATALOG":
            continue
        for schema in client.schemas.list(catalog_name=catalog.name, include_browse=True):
            for table in client.tables.list(
                catalog_name=catalog.name,
                schema_name=schema.name,
                include_browse=True,
                omit_columns=True,
                omit_properties=False,
            ):
                source = (table.properties or {}).get("source_table", "").casefold()
                if source in found:
                    found[source].append(table.full_name)
    return found


def _validate_existing_publication(
    client: Any,
    spec: OnlinePublicationSpec,
    lookup: FeatureTrainingSpec,
    primary_keys: tuple[str, ...],
) -> None:
    """Keep UC identities separate from Delta identities and inspect native routing."""
    source = client.tables.get(spec.source_table)
    table = client.tables.get(spec.online_table)
    properties = table.properties or {}
    if table.browse_only is True or (
        table.full_name,
        properties.get("source_table"),
        properties.get("source_table_id"),
    ) != (spec.online_table, spec.source_table, source.table_id):
        raise ValueError("Publication UC source identity or read access differs from its binding.")
    synced = client.database.get_synced_database_table(name=spec.online_table)
    if (
        synced.name,
        synced.effective_database_instance_name,
        synced.spec.source_table_full_name,
    ) != (spec.online_table, spec.online_store, spec.source_table):
        raise ValueError("Publication source or effective online store differs from its binding.")
    _validate_publication_keys(synced, lookup, spec.source_table, primary_keys)
    _require_first_sync(synced, table.pipeline_id)


def _validate_publication_keys(
    synced: Any, lookup: FeatureTrainingSpec, source: str, primary_keys: tuple[str, ...]
) -> None:
    """Check feature-table keys while preserving positional request-column mappings."""
    keys = tuple(synced.spec.primary_key_columns or ())
    if (
        synced.spec.timeseries_key
        or keys != primary_keys
        or any(
            item.timestamp_columns or len(item.lookup_key) != len(keys)
            for item in lookup.lookups
            if item.table_name == source
        )
    ):
        raise ValueError("Publication primary keys differ from the saved non-temporal lookup.")


def _require_first_sync(synced: Any, pipeline_id: str | None) -> None:
    """Reject provisioning or failed copies even when their UC metadata already exists."""
    provision = synced.unity_catalog_provisioning_state
    status = synced.data_synchronization_status
    state = getattr(status, "detailed_state", None)
    online = {
        "SYNCED_TABLE_ONLINE",
        "SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE",
        "SYNCED_TABLE_ONLINE_TRIGGERED_UPDATE",
        "SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE",
    }
    if (
        getattr(provision, "value", provision) != "ACTIVE"
        or getattr(state, "value", state) not in online
    ):
        raise ValueError("Publication is not active with a healthy online synchronization state.")
    if (
        not pipeline_id
        or status.pipeline_id != pipeline_id
        or not getattr(status.last_sync, "sync_end_timestamp", None)
    ):
        raise ValueError("Publication requires an inspectable pipeline and a completed first sync.")


def publish_online_features(
    spark: Any,
    spec: OnlinePublicationSpec,
    *,
    admission: PublishAdmission,
    feature_client: Any = None,
) -> dict[str, Any]:
    """Validate and request one native TRIGGERED sync without claiming completion.

    The caller must serialize publication and source writers with the same
    admission authority and restrict external writes. The SDK publishes latest
    data, not a versionAsOf snapshot. The recorded source snapshot is the checked
    preflight version, not an attested online version. Native sync is repeatable;
    this adapter does not provide an exactly-once request token or retry after an
    ambiguous failure. Inspect the native pipeline before retrying such a call.
    """
    if not isinstance(spec, OnlinePublicationSpec):
        raise TypeError("spec must be OnlinePublicationSpec.")
    if getattr(admission, "local_only", True) is not False or not callable(
        getattr(admission, "hold", None)
    ):
        raise ValueError("Online publication requires explicit shared writer admission.")
    client = feature_engineering_client(feature_client)
    with admission.hold(spec.source_table_id):
        store = _available_store(client, spec.online_store)
        snapshot = _source_contract(spark, spec, client)
        if snapshot_table(spark, spec.source_table) != snapshot:
            raise ValueError("Online feature source changed during preflight; retry validation.")
        submitted_at = time.time_ns() // 1_000_000
        published = client.publish_table(
            online_store=store,
            source_table_name=spec.source_table,
            online_table_name=spec.online_table,
            publish_mode="TRIGGERED",
        )
        if (
            getattr(published, "online_table_name", None) != spec.online_table
            or not isinstance(getattr(published, "pipeline_id", None), str)
            or not published.pipeline_id
        ):
            raise ValueError("Native publication returned no matching table and pipeline receipt.")
        return {
            "spec": asdict(spec),
            "source_snapshot": snapshot,
            "pipeline_id": published.pipeline_id,
            "submitted_at_ms": submitted_at,
            "status": "SUBMITTED",
        }


def _publication_identity(receipt: dict[str, Any]) -> tuple[str, int]:
    """Validate the recorded native sync selector before polling its updates."""
    OnlinePublicationSpec(**receipt["spec"])
    pipeline, started = receipt["pipeline_id"], receipt["submitted_at_ms"]
    if not isinstance(pipeline, str) or not pipeline or type(started) is not int or started < 0:
        raise ValueError("Online publication receipt has no valid pipeline or submission time.")
    return pipeline, started


def online_publication_status(client: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    """Inspect a new native pipeline update without blocking or triggering another sync.

    SYNCED means a pipeline update started after submission has completed under
    the caller's single-writer ownership. It is not proof of any entity's feature
    age or of a particular source version. The saved model's online policy still
    checks every request. Retain the returned update ID for operator inspection.
    """
    pipeline, started = _publication_identity(receipt)
    updates = client.pipelines.list_updates(pipeline_id=pipeline, max_results=1).updates or []
    if not updates:
        return {**receipt, "status": "PENDING"}
    update = updates[0]
    if update.pipeline_id != pipeline:
        raise ValueError("Native update belongs to a different publication pipeline.")
    if update.creation_time is None or update.creation_time < started:
        return {**receipt, "status": "PENDING"}
    if not isinstance(update.update_id, str) or not update.update_id:
        raise ValueError("Native publication update has no inspectable update ID.")
    state = getattr(update.state, "value", update.state)
    status = {"COMPLETED": "SYNCED", "FAILED": "FAILED", "CANCELED": "FAILED"}.get(state, "PENDING")
    return {**receipt, "status": status, "update_id": update.update_id, "update_state": state}
