"""Exercise pinned recovery through real local scoring and an in-memory Delta boundary."""

import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from tests.integration.platforms.test_cdf_recovery import StructuredError

from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.integrations.databricks.data.admission import BatchConflictError
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
from skyulf.integrations.databricks.scoring.incremental import local_incremental as batch
from skyulf.integrations.databricks.scoring.local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightResult,
    PreparedLocalWorkflow,
)


class Frame:
    """Implement only Spark transport operations used by the local scoring boundary."""

    def __init__(self, data, types, owner):
        """Retain explicit Spark types, including for an empty frame."""
        self.data = data.copy()
        self.types = types
        self.owner = owner
        self.columns = list(data.columns)
        self.schema = Schema(types)

    def select(self, *columns):
        """Project before decoding rows or constructing the typed prediction bridge."""
        return Frame(
            self.data.loc[:, list(columns)], {key: self.types[key] for key in columns}, self.owner
        )

    def orderBy(self, *columns):
        """Use stable row key ordering for the real bounded driver collection."""
        return Frame(self.data.sort_values(list(columns)), self.types, self.owner)

    def limit(self, count):
        """Bound the rows evaluated by guards and collection."""
        return Frame(self.data.iloc[:count], self.types, self.owner)

    def count(self):
        """Expose materialized row counts without bypassing membership checks."""
        return len(self.data)

    def where(self, predicate):
        """Evaluate the real CDF correction and timestamp guards on local rows."""
        return Frame(self.data.loc[predicate(self.data)], self.types, self.owner)

    def toLocalIterator(self):
        """Decode driver records using the same interface as PySpark Row."""
        for record in self.data.to_dict("records"):
            yield SimpleNamespace(asDict=lambda recursive, record=record: record)

    def join(self, other, on, how):
        """Support the writer's collision check and event-time membership join."""
        data = self.data.merge(other.data, on=on, how="inner")
        if how == "left_semi":
            data = data.loc[:, self.columns]
        return Frame(data, self.types | other.types, self.owner)

    def withColumn(self, name, value):
        """Attach constant publication provenance to each scored row."""
        return Frame(self.data.assign(**{name: value}), self.types | {name: "string"}, self.owner)

    @property
    def write(self):
        """Open a writer that changes committed state only at saveAsTable."""
        return Writer(self)


@dataclass(frozen=True)
class DataType:
    """Compare Spark field types by value instead of object identity."""

    name: str

    def typeName(self):
        """Expose the Spark primitive name consumed by schema guards."""
        return self.name


class Schema:
    """Expose indexed and iterable Spark-style schema fields."""

    def __init__(self, types):
        """Use shared type strings as equality tokens and supply typeName access."""
        self.types = types

    def __getitem__(self, name):
        """Describe one declared field for target compatibility validation."""
        return SimpleNamespace(dataType=DataType(self.types[name]))

    def __iter__(self):
        """Compare field names and types in final output validation."""
        return iter(
            SimpleNamespace(name=name, dataType=value) for name, value in self.types.items()
        )


class Column:
    """Evaluate basic Spark column predicates at the transport boundary."""

    def __init__(self, name):
        """Keep the requested source control column."""
        self.name = name

    def __eq__(self, value):
        """Select insert changes without replacing production CDF validation."""
        return lambda frame: frame[self.name].eq(value)

    def __ne__(self, value):
        """Identify updates and deletes for the production rejection guard."""
        return lambda frame: frame[self.name].ne(value)

    def isNull(self):
        """Identify missing event timestamps."""
        return lambda frame: frame[self.name].isna()


class Writer:
    """Store the single atomic writer operation and its JSON receipt."""

    def __init__(self, frame):
        """Start with append semantics matching the DataFrameWriter default."""
        self.frame = frame
        self.options = {}
        self.write_mode = "append"

    def format(self, name):
        """Accept the verified Delta provider selection."""
        assert name == "delta"
        return self

    def mode(self, mode):
        """Record whether existing rows survive the transaction."""
        self.write_mode = mode
        return self

    def option(self, key, value):
        """Retain the production transaction and schema options for assertions."""
        self.options[key] = value
        return self

    def saveAsTable(self, target):
        """Atomically replace or append both prediction rows and their receipt."""
        owner = self.frame.owner
        assert owner.held and target == "db.target"
        if owner.write_failure:
            raise RuntimeError("injected Delta write failure")
        data = self.frame.data
        if self.write_mode == "append" and not owner.target.data.empty:
            data = pd.concat([owner.target.data, data], ignore_index=True)
        owner.target = Frame(data, self.frame.types, owner)
        owner.target_version += 1
        owner.previous = json.loads(self.options["userMetadata"])
        owner.writes.append((self.write_mode, self.options))


class Reader:
    """Resolve immutable source snapshots and expose CDF history loss explicitly."""

    def __init__(self, owner):
        """Collect versionAsOf independently for every new Spark reader."""
        self.owner = owner
        self.options = {}

    def format(self, name):
        """Keep the reader in the Delta format."""
        assert name == "delta"
        return self

    def option(self, key, value):
        """Record the exact source window requested by production code."""
        self.options[key] = value
        return self

    def table(self, name):
        """Fail expired change feed reads while retaining the pinned full snapshot."""
        assert name == "db.source"
        self.owner.reads.append(dict(self.options))
        if "readChangeFeed" in self.options:
            if self.owner.read_failure is not None:
                raise self.owner.read_failure
            return Frame(
                self.owner.cdf, self.owner.source_types | {"_change_type": "string"}, self.owner
            )
        version = self.options["versionAsOf"]
        return Frame(self.owner.snapshots[version], self.owner.source_types, self.owner)


class Store:
    """Represent retained snapshots, stable table IDs and admitted Delta publications."""

    local_only = False

    def __init__(self, prepared):
        """Start after one trusted publication with an unavailable later CDF interval."""
        self.prepared = prepared
        self.source_types = {"id": "long", "x": "double"}
        self.snapshots = {10: pd.DataFrame({"id": [1, 2], "x": [3.0, 5.0]})}
        types = {
            "id": "long",
            "prediction": "double",
            "run_id": "string",
            "model_name": "string",
            "model_version": "string",
        }
        self.target = Frame(
            pd.DataFrame(
                {
                    "id": [1],
                    "prediction": [-1.0],
                    "run_id": ["old"],
                    "model_name": ["db.model"],
                    "model_version": ["2"],
                }
            ),
            types,
            self,
        )
        self.ids = {"db.source": "source-id", "db.target": "target-id"}
        self.previous = {
            "skyulf_mode": "incremental_append",
            "source_table_id": "source-id",
            "target_table_id": "target-id",
            "source_end_version": 7,
            "model_name": "db.model",
            "model_version": "2",
            "model_digest": prepared.preflight.model_digest,
        }
        self.target_version = 4
        self.source_version = 10
        self.reads = []
        self.writes = []
        self.held = False
        self.write_failure = False
        self.read_failure = StructuredError("DELTA_TRUNCATED_TRANSACTION_LOG")
        self.cdf = pd.DataFrame()

    @property
    def read(self):
        """Supply a fresh read builder to prevent leaked version options."""
        return Reader(self)

    def table(self, name):
        """Return the latest committed target only."""
        assert name == "db.target"
        return self.target

    def createDataFrame(self, rows, schema):
        """Materialize the actual predictions under the existing target schema."""
        return Frame(pd.DataFrame(rows, columns=list(schema.types)), schema.types, self)

    @contextmanager
    def hold(self, table_id):
        """Expose admission lifetime to every commit operation."""
        assert table_id == self.ids["db.target"]
        self.held = True
        try:
            yield
        finally:
            self.held = False


@pytest.fixture
def recovery_case(tmp_path, monkeypatch):
    """Keep real artifacts and scoring while replacing only the unavailable Spark transport."""
    data = pd.DataFrame({"x": np.arange(20, dtype=float), "target": np.arange(20, dtype=float) * 2})
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=data.iloc[:16], test=data.iloc[16:]),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=20,
        max_bytes=10000,
    )
    config = LocalWorkflowConfig(
        runtime="databricks",
        engine="pandas",
        source=InputSource(
            kind="uc_table",
            table="db.source",
            read_mode="incremental",
            max_rows=10,
            max_bytes=10000,
        ),
        model=ModelSelection(kind="local_pipeline", name="db.model", version="2"),
        sink=OutputSink(kind="uc_delta", table="db.target"),
    )
    prepared = PreparedLocalWorkflow(
        config,
        artifact,
        PreflightResult(
            issues=(),
            remote_checked=True,
            model_version="2",
            model_digest=artifact.manifest.pipeline_sha256,
            output_schema=(ColumnSpec(name="prediction", dtype="float64"),),
        ),
    )
    store = Store(prepared)
    monkeypatch.setattr(batch, "table_identity", lambda spark, name: store.ids[name])
    monkeypatch.setattr(batch, "require_incremental_change_feed", lambda *args: None)
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: {
            "version": store.source_version if name == "db.source" else store.target_version,
            "userMetadata": None if name == "db.source" else json.dumps(store.previous),
        },
    )
    original_import = batch.importlib.import_module
    monkeypatch.setattr(
        batch.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(lit=lambda value: value, col=Column)
            if name == "pyspark.sql.functions"
            else original_import(name)
        ),
    )
    return store


def recovery_request(store):
    """Obtain the routed request from the real incremental execution boundary."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import CdfRecoveryRequired

    with pytest.raises(CdfRecoveryRequired) as failure:
        batch.run_incremental_local_batch(
            store, store.prepared, record_key_columns=("id",), admission=store
        )
    return failure.value.request


def recover(store, request):
    """Use the explicit recovery entry on the same public scoring function."""
    return batch.run_incremental_local_batch(
        store,
        store.prepared,
        record_key_columns=("id",),
        admission=store,
        recovery_request=request,
    )


def test_full_recovery_is_pinned_atomic_replayable_and_resumes(recovery_case):
    """An advanced source must not change an already routed recovery snapshot."""
    store = recovery_case
    request = recovery_request(store)
    assert request["source_start_version"] == 7 and request["source_end_version"] == 10
    assert store.writes == []
    store.source_version = 11
    store.snapshots[11] = pd.DataFrame({"id": [1, 2, 3], "x": [3.0, 5.0, 7.0]})
    result = recover(store, request)
    assert not result.noop and result.input_count == result.output_count == 2
    assert store.target.data.prediction.tolist() == pytest.approx([6.0, 10.0])
    assert store.target.data.id.tolist() == [1, 2]
    assert store.reads[-1] == {"versionAsOf": 10}
    assert len(store.writes) == 1 and store.writes[0][0] == "overwrite"
    assert store.writes[0][1]["partitionOverwriteMode"] == "static"
    assert store.previous["cdf_recovered"] is True
    replay = recover(store, request)
    assert replay.noop and replay.commit_version == result.commit_version
    assert len(store.writes) == 1
    store.source_version = 10
    ordinary = batch.run_incremental_local_batch(
        store, store.prepared, record_key_columns=("id",), admission=store
    )
    assert ordinary.noop and ordinary.source_end_version == 10


@pytest.mark.parametrize("change", ["source_id", "target_id", "target_commit", "model", "artifact"])
def test_stale_recovery_does_not_touch_prior_predictions(recovery_case, change):
    """Changed identities or a foreign commit must fail before reading and overwriting output."""
    store = recovery_case
    request = recovery_request(store)
    if change.endswith("_id"):
        store.ids["db.source" if change == "source_id" else "db.target"] = "replacement"
    elif change == "target_commit":
        store.target_version += 1
    elif change == "model":
        store.prepared = replace(
            store.prepared,
            config=store.prepared.config.model_copy(
                update={
                    "model": ModelSelection(kind="local_pipeline", name="db.model", version="3")
                }
            ),
        )
        store.prepared = replace(
            store.prepared, preflight=replace(store.prepared.preflight, model_version="3")
        )
    else:
        request["model_digest"] = "b" * 64
    with pytest.raises(BatchConflictError):
        recover(store, request)
    assert store.target.data.prediction.tolist() == [-1.0] and store.writes == []


@pytest.mark.parametrize("budget", ["max_rows", "max_bytes"])
def test_full_snapshot_budgets_abort_before_replacement(recovery_case, budget):
    """Recovery must obey the whole-snapshot driver budget instead of truncating results."""
    store = recovery_case
    request = recovery_request(store)
    config = store.prepared.config
    store.prepared = replace(
        store.prepared,
        config=config.model_copy(update={"source": config.source.model_copy(update={budget: 1})}),
    )
    with pytest.raises(ValueError, match=budget):
        recover(store, request)
    assert store.writes == [] and store.target_version == 4


def test_empty_full_snapshot_clears_predictions_and_commits_progress(recovery_case):
    """An empty retained snapshot must remove stale output rather than returning a no-op."""
    store = recovery_case
    request = recovery_request(store)
    store.snapshots[10] = store.snapshots[10].iloc[:0]
    result = recover(store, request)
    assert not result.noop and result.output_count == 0
    assert store.target.data.empty and store.previous["source_end_version"] == 10


def test_failed_recovery_write_keeps_prior_rows_and_receipt(recovery_case):
    """A retry after a failed transaction must use the unchanged pinned base state."""
    from skyulf.integrations.databricks.data.delta_io.delta import DeltaPublishError

    store = recovery_case
    request = recovery_request(store)
    previous = dict(store.previous)
    store.write_failure = True
    with pytest.raises(DeltaPublishError):
        recover(store, request)
    assert store.previous == previous and store.target.data.prediction.tolist() == [-1.0]
    store.write_failure = False
    assert recover(store, request).output_count == 2


def _use_temporal_model(store, tmp_path):
    """Load a real carry model whose full recovery must discard earlier committed history."""
    from tests.integration.platforms.test_local_temporal_history import fitted_temporal_pipeline

    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline

    save_local_pipeline(fitted_temporal_pipeline(), tmp_path / "temporal")
    artifact = load_local_pipeline(tmp_path / "temporal")
    store.prepared = replace(
        store.prepared,
        artifact=artifact,
        preflight=replace(
            store.prepared.preflight,
            model_digest=artifact.manifest.pipeline_sha256,
        ),
    )
    store.source_types = {"id": "long", "t": "long", "v": "double"}
    store.snapshots[10] = pd.DataFrame({"id": [0, 1, 2], "t": [0, 1, 2], "v": [12.0, 1.0, 2.0]})
    return artifact


@pytest.mark.parametrize("empty", [False, True])
def test_temporal_recovery_rebuilds_carry_and_next_append(recovery_case, tmp_path, empty):
    """Full recovery must match fresh causal scoring and preserve the next increment's tail."""
    from skyulf.integrations.databricks.scoring.incremental.local_history import incremental_history

    store = recovery_case
    _use_temporal_model(store, tmp_path)
    if empty:
        store.snapshots[10] = store.snapshots[10].iloc[:0]
    request = recovery_request(store)
    result = recover(store, request)
    assert result.manifest is not None
    saved = result.manifest["temporal_history"]
    if empty:
        assert all(rows == [] for rows in saved["steps"].values())
    else:
        with incremental_history(store.prepared, None) as fresh:
            expected = store.prepared.predict(store.snapshots[10][["t", "v"]])
        assert store.target.data.prediction.tolist() == pytest.approx(expected.prediction.tolist())
        assert saved == fresh.state
    store.source_version = 11
    store.read_failure = None
    store.cdf = pd.DataFrame({"id": [3], "t": [3], "v": [3.0], "_change_type": ["insert"]})
    continued = batch.run_incremental_local_batch(
        store,
        store.prepared,
        record_key_columns=("id",),
        admission=store,
    )
    whole = pd.concat([store.snapshots[10], store.cdf.drop(columns="_change_type")])
    with incremental_history(store.prepared, None) as fresh:
        expected = store.prepared.predict(whole[["t", "v"]])
    assert continued.output_count == 1 and continued.manifest is not None
    assert continued.manifest["temporal_history"] == fresh.state
    assert store.target.data.prediction.tolist() == pytest.approx(expected.prediction.tolist())


def test_late_cdf_materialization_loss_emits_pinned_request(recovery_case, monkeypatch):
    """Lazy executor failures after source selection still need the exact admitted request."""
    store = recovery_case
    store.read_failure = None
    store.cdf = pd.DataFrame({"id": [2], "x": [5.0], "_change_type": ["insert"]})

    def fail_iterator(self):
        """Represent expiry discovered only during bounded driver collection."""
        raise StructuredError("DELTA_CHANGE_DATA_FILE_NOT_FOUND")

    monkeypatch.setattr(Frame, "toLocalIterator", fail_iterator)
    request = recovery_request(store)
    assert request["source_end_version"] == 10 and store.writes == []


@pytest.mark.parametrize(
    "condition",
    [
        "DELTA_MISSING_CHANGE_DATA",
        "DELTA_FILE_NOT_FOUND",
        "INSUFFICIENT_PERMISSIONS",
        "NETWORK_ERROR",
    ],
)
def test_nonhistory_source_errors_never_create_recovery(recovery_case, condition):
    """The publication path must preserve unrelated remote failure types without a write."""
    store = recovery_case
    store.read_failure = StructuredError(condition)
    with pytest.raises(StructuredError):
        batch.run_incremental_local_batch(
            store, store.prepared, record_key_columns=("id",), admission=store
        )
    assert store.writes == [] and store.target_version == 4


def test_readable_source_correction_still_requires_existing_policy(recovery_case):
    """An UPDATE or DELETE must not be relabeled as expired CDF just to enable recovery."""
    store = recovery_case
    store.read_failure = None
    store.cdf = pd.DataFrame({"id": [1], "x": [5.0], "_change_type": ["delete"]})
    with pytest.raises(batch.SourceChangeRequiresRebuild):
        batch.run_incremental_local_batch(
            store, store.prepared, record_key_columns=("id",), admission=store
        )
    assert store.writes == [] and store.previous["source_end_version"] == 7
