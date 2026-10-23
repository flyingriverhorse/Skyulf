"""Feature job repairs preserve snapshot identity and explicit source ownership."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from skyulf.integrations.databricks.features import delta, runtime
from skyulf.integrations.databricks.features.config import FeatureGroup, FeaturePlan
from skyulf.integrations.databricks.features.runtime import selected_groups, transform_source


def _plan():
    """A selected group has one project-contained transform and output table."""
    group = FeatureGroup(
        "company", "raw", "features", "src/features/groups/company.py:compute", ("size",)
    )
    return FeaturePlan("base", "merged", ("id",), "at", (group,))


def test_selective_run_has_explicit_subset():
    """Typos must not silently skip a required feature group."""
    assert selected_groups(_plan(), "*") == {"company"}
    assert selected_groups(_plan(), "company") == {"company"}
    assert selected_groups(_plan(), "") == set()
    with pytest.raises(ValueError, match="Unknown"):
        selected_groups(_plan(), "comapny")


def test_transform_source_is_read_without_execution(tmp_path):
    """Initialize must pin project code without triggering a transformation early."""
    path = tmp_path / "src/features/groups/company.py"
    path.parent.mkdir(parents=True)
    path.write_text("raise AssertionError('not now')\n")
    source, digest = transform_source(tmp_path, _plan().groups[0])
    assert source == path.read_bytes().decode("utf-8")
    assert len(digest) == 64


def test_transform_path_cannot_escape_project(tmp_path):
    """Even manually constructed plans must not load source outside the project root."""
    group = FeatureGroup("bad", "raw", "out", "../../outside.py:compute", ("size",))
    with pytest.raises(ValueError, match="inside"):
        transform_source(Path(tmp_path), group)


class _FeatureFrame:
    """Keep the publication tests focused on Delta metadata and observable values."""

    columns = ["id", "at", "size"]
    schema = SimpleNamespace(fields=[SimpleNamespace(name=name, dataType=name) for name in columns])

    def __init__(self, spark, value):
        """Bind a single feature value to its simulated Spark session."""
        self.spark, self.value = spark, value

    def select(self, *columns):
        """Preserve the already selected feature shape."""
        return self

    def createOrReplaceTempView(self, name):
        """Expose the batch to the actual publisher's SQL construction."""
        self.spark.views[name] = self


class _FeatureSpark:
    """Model atomic metadata/row commits and fault windows without a Spark service."""

    def __init__(self):
        """Start with empty tables and independently controllable failure windows."""
        self.tables, self.views, self.operations = {}, {}, []
        self.catalog = SimpleNamespace(
            tableExists=lambda table: table in self.tables,
            dropTempView=lambda name: self.views.pop(name),
        )
        self.read = self
        self.version = None
        self.fail_merge = False
        self.lose_response = None

    def put(self, table, value, *, properties=None):
        """Atomically create a table with rows and optional publication properties."""
        self.tables[table] = {
            "id": f"id-{table}",
            "version": 0,
            "properties": properties or {"delta.enableChangeDataFeed": "true"},
            "values": [value],
        }

    def format(self, name):
        """Accept the pinned Delta read format."""
        return self

    def option(self, name, value):
        """Capture the exact version requested for the next table read."""
        self.version = value
        return self

    def table(self, table):
        """Read either the current value or the requested historical version."""
        state = self.tables[table]
        version, self.version = self.version, None
        return _FeatureFrame(
            self, state["values"][state["version"] if version is None else version]
        )

    def sql(self, statement):
        """Apply each supported metadata or data write as one atomic commit."""
        import json
        import re

        self.operations.append(statement)
        table = statement.split()[2].replace("`", "")
        if statement.startswith("DESCRIBE DETAIL"):
            state = self.tables[table]
            row = {"id": state["id"], "format": "delta", "properties": state["properties"]}
            return SimpleNamespace(first=lambda: row)
        if statement.startswith("DESCRIBE HISTORY"):
            row = [self.tables[table]["version"]]
            return SimpleNamespace(select=lambda name: SimpleNamespace(first=lambda: row))
        if statement.startswith("ALTER TABLE"):
            state = self.tables[table]
            key, value = re.findall(r"'([^']*)'\s*=\s*'([^']*)'", statement)[0]
            state["properties"][key] = value
            state["version"] += 1
            state["values"].append(state["values"][-1])
            if self.lose_response == "intent":
                self.lose_response = None
                raise RuntimeError("lost intent response")
            return None
        if statement.startswith("CREATE TABLE"):
            pairs = dict(re.findall(r"'([^']*)'\s*=\s*'([^']*)'", statement))
            assert any(
                json.loads(value) for key, value in pairs.items() if key.startswith("skyulf.")
            )
            self.put(table, self.views[statement.rsplit("`", 2)[1]].value, properties=pairs)
            if self.lose_response == "create":
                self.lose_response = None
                raise RuntimeError("lost create response")
            return None
        if statement.startswith("MERGE INTO"):
            if self.fail_merge:
                raise RuntimeError("transport interrupted before MERGE")
            state = self.tables[table]
            value = self.views[statement.split(" USING `")[1].split("`")[0]].value
            if value != state["values"][-1]:
                state["version"] += 1
                state["values"].append(value)
            if self.lose_response == "merge":
                self.lose_response = None
                raise RuntimeError("lost merge response")
            return None
        raise AssertionError(statement)


@pytest.fixture
def feature_publication(monkeypatch, tmp_path):
    """A deterministic transform makes safe receipt-loss replay observable."""
    spark = _FeatureSpark()
    for table, value in (("catalog.features.base", 1), ("catalog.features.raw", 10)):
        spark.put(table, value)
    group = FeatureGroup(
        "company",
        "catalog.features.raw",
        "catalog.features.group",
        "src/features/groups/company.py:compute",
        ("size",),
    )
    plan = FeaturePlan("catalog.features.base", "catalog.features.merged", ("id",), "at", (group,))
    spark.put(group.output_table, 10)
    spark.put(plan.output_table, 10)
    path = tmp_path / "src/features/groups/company.py"
    path.parent.mkdir(parents=True)
    path.write_text("def compute(frame):\n    return frame\n")
    monkeypatch.setattr(delta, "validate_feature_frame", lambda *args: 1)
    monkeypatch.setattr(
        runtime, "join_feature_groups", lambda base, frames, plan: frames["company"]
    )
    return spark, plan, tmp_path


def _build(feature_publication, snapshot):
    """Return a real runtime receipt backed by the in-memory Delta commit model."""
    spark, plan, project = feature_publication
    return runtime.build_feature_group(spark, project, plan, snapshot, "company")


def test_feature_snapshot_pins_output_identity_and_code_only_generation(feature_publication):
    """New code with unchanged source versions must still supersede older repairs."""
    spark, plan, project = feature_publication
    first = runtime.initialize_features(spark, project, plan, "*")
    _build(feature_publication, first)
    (project / "src/features/groups/company.py").write_text(
        "def compute(frame):\n    # Changed implementation, unchanged source.\n    return frame\n"
    )
    second = runtime.initialize_features(spark, project, plan, "*")
    assert second["publication"]["generation"] == first["publication"]["generation"] + 1
    assert first["source_versions"] == second["source_versions"]
    assert first["transform_sha256"] != second["transform_sha256"]
    assert (
        second["publication"]["outputs"][plan.groups[0].output_table]["table_id"]
        == (spark.tables[plan.groups[0].output_table]["id"])
    )


@pytest.mark.parametrize("phase", ["group", "merge"])
def test_old_feature_repair_rejects_newer_partial_group(feature_publication, phase):
    """A newer group must fence both stale group tasks and the final merged output."""
    spark, plan, project = feature_publication
    old = runtime.initialize_features(spark, project, plan, "*")
    receipt = _build(feature_publication, old)
    runtime.merge_feature_groups(spark, plan, old, {"company": receipt})
    newer = runtime.initialize_features(spark, project, plan, "*")
    _build(feature_publication, newer)
    before = list(spark.operations)
    with pytest.raises(ValueError, match="superseded"):
        if phase == "group":
            _build(feature_publication, old)
        else:
            runtime.merge_feature_groups(spark, plan, old, {"company": receipt})
    assert not any(
        sql.startswith(("MERGE", "ALTER", "CREATE")) for sql in spark.operations[len(before) :]
    )


@pytest.mark.parametrize("lost_receipt", [False, True])
def test_same_feature_snapshot_retries_noop_and_lost_receipt(feature_publication, lost_receipt):
    """A committed intent can retry without erasing the newer-generation fence."""
    spark, plan, project = feature_publication
    initial = runtime.initialize_features(spark, project, plan, "*")
    _build(feature_publication, initial)
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    spark.fail_merge = not lost_receipt
    if spark.fail_merge:
        with pytest.raises(RuntimeError, match="interrupted"):
            _build(feature_publication, snapshot)
    else:
        _build(feature_publication, snapshot)  # Successful write whose task receipt was lost.
    spark.fail_merge = False
    receipt = _build(feature_publication, snapshot)
    assert spark.tables[receipt["table"]]["values"][-1] == 10
    assert receipt["snapshot_sha256"] == runtime.snapshot_digest(snapshot)


def test_feature_repair_rejects_recreated_table_at_same_version(feature_publication):
    """A replacement table cannot inherit publication authority from its predecessor."""
    spark, plan, project = feature_publication
    initial = runtime.initialize_features(spark, project, plan, "*")
    _build(feature_publication, initial)
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    spark.tables[plan.groups[0].output_table]["id"] = "replacement-id"
    with pytest.raises(ValueError, match="identity"):
        _build(feature_publication, snapshot)
    assert spark.tables[plan.groups[0].output_table]["values"][-1] == 10


def test_unguarded_feature_write_refuses_managed_output(feature_publication):
    """The low-level writer must not bypass an established job publication fence."""
    spark, plan, project = feature_publication
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    _build(feature_publication, snapshot)
    with pytest.raises(ValueError, match="publication"):
        delta.publish_feature_table(
            spark, _FeatureFrame(spark, 999), plan.groups[0].output_table, plan.keys, plan.timestamp
        )
    assert spark.tables[plan.groups[0].output_table]["values"][-1] == 10


def test_legacy_feature_snapshot_requires_reinitialization_before_mutation(feature_publication):
    """Saved pre-fence snapshots cannot establish which newer writes they may replace."""
    spark, plan, project = feature_publication
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    snapshot.pop("publication", None)
    with pytest.raises(ValueError, match="publication"):
        _build(feature_publication, snapshot)
    assert not any(sql.startswith(("MERGE", "ALTER", "CREATE")) for sql in spark.operations)


@pytest.mark.parametrize("failure", ["create", "intent", "merge"])
def test_feature_publication_recovers_lost_write_responses(feature_publication, failure):
    """Intent, first-create and MERGE response loss must all preserve safe same-run retry."""
    spark, plan, project = feature_publication
    output = plan.groups[0].output_table
    if failure == "create":
        del spark.tables[output]
    spark.tables[plan.groups[0].source_table]["values"][0] = 42
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    spark.lose_response = failure
    with pytest.raises(RuntimeError, match="lost"):
        _build(feature_publication, snapshot)
    marker = delta.publication_state(spark, output)
    assert marker is not None
    assert marker["intent"]["snapshot_sha256"] == runtime.snapshot_digest(snapshot)
    receipt = _build(feature_publication, snapshot)
    assert spark.tables[output]["values"][-1] == 42
    assert delta.read_version(spark, output, receipt["version"]).value == 42
    assert len([sql for sql in spark.operations if sql.startswith("CREATE")]) == (
        1 if failure == "create" else 0
    )


def test_feature_initial_create_commits_rows_and_fence_together(feature_publication):
    """A first publication must never expose unfenced rows before an ALTER receipt."""
    spark, plan, project = feature_publication
    del spark.tables[plan.groups[0].output_table]
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    receipt = _build(feature_publication, snapshot)
    writes = [sql for sql in spark.operations if sql.startswith(("ALTER", "CREATE", "MERGE"))]
    assert len(writes) == 1 and writes[0].startswith("CREATE TABLE")
    assert "skyulf.features.publication" in writes[0]
    assert delta.read_version(spark, receipt["table"], receipt["version"]).value == 10


@pytest.mark.parametrize(
    "raw",
    [
        "invalid-json",
        "x" * 257,
        "{}",
        '{"generation":true,"snapshot_sha256":"' + "a" * 64 + '"}',
        '{"generation":1,"snapshot_sha256":"invalid"}',
    ],
)
def test_feature_publication_rejects_corrupt_marker(feature_publication, raw):
    """Unknown publication metadata cannot silently downgrade a managed output."""
    spark, plan, project = feature_publication
    spark.tables[plan.groups[0].output_table]["properties"]["skyulf.features.publication"] = raw
    with pytest.raises(ValueError):
        runtime.initialize_features(spark, project, plan, "*")
    assert not any(sql.startswith(("MERGE", "ALTER", "CREATE")) for sql in spark.operations)


def test_feature_publication_refuses_changed_pinned_head(feature_publication):
    """An unmanaged intervening write must not receive authority from a stale initialization."""
    spark, plan, project = feature_publication
    snapshot = runtime.initialize_features(spark, project, plan, "*")
    state = spark.tables[plan.groups[0].output_table]
    state["version"] += 1
    state["values"].append(999)
    with pytest.raises(ValueError, match="head changed"):
        _build(feature_publication, snapshot)
    assert state["values"][-1] == 999
    assert "skyulf.features.publication" not in state["properties"]


def test_feature_publication_refuses_competing_same_generation(feature_publication):
    """Distinct initializations before the first claim cannot repair over each other."""
    spark, plan, project = feature_publication
    first = runtime.initialize_features(spark, project, plan, "*")
    second = runtime.initialize_features(spark, project, plan, "*")
    _build(feature_publication, first)
    with pytest.raises(ValueError, match="superseded"):
        _build(feature_publication, second)
    assert first["publication"]["generation"] == second["publication"]["generation"]
    assert first["publication"]["snapshot_id"] != second["publication"]["snapshot_id"]


def test_legacy_feature_snapshot_can_read_reuse_but_cannot_publish(feature_publication):
    """Read-only reuse remains possible while unfenced final writes require reinitialization."""
    spark, plan, project = feature_publication
    snapshot = runtime.initialize_features(spark, project, plan, "")
    snapshot.pop("publication")
    receipt = _build(feature_publication, snapshot)
    with pytest.raises(ValueError, match="publication"):
        runtime.merge_feature_groups(spark, plan, snapshot, {"company": receipt})
    assert receipt["reused"] is True
    assert not any(sql.startswith(("MERGE", "ALTER", "CREATE")) for sql in spark.operations)


def _base_plan():
    """A raw base whose target and key come from one project function, without groups."""
    return FeaturePlan(
        "c.s.raw", "c.s.merged", ("id", "day"), "at", (), "src/features/base.py:prepare"
    )


def _base_project(tmp_path, body="    return [*frame, 'derived']\n"):
    """Write the project's base transform and return its root."""
    path = tmp_path / "src/features/base.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def prepare(frame):\n" + body, encoding="utf-8")
    return tmp_path


def _merge_spies(monkeypatch):
    """Replace Delta reads, joins and publication with recorders."""
    seen = {}
    monkeypatch.setattr(runtime, "_require_snapshot", lambda plan, snapshot: None)
    monkeypatch.setattr(runtime, "_check_publication", lambda *args: {})
    monkeypatch.setattr(runtime, "read_version", lambda spark, table, version: ["raw-row"])
    monkeypatch.setattr(
        runtime, "join_feature_groups", lambda base, frames, plan: seen.setdefault("base", base)
    )
    monkeypatch.setattr(runtime, "publish_feature_table", lambda *args, **kwargs: {"version": 2})
    return seen


def test_base_transform_runs_on_the_pinned_base_before_joins(tmp_path, monkeypatch):
    """Labels and keys derived by prepare() must reach the published observation table."""
    project = _base_project(tmp_path)
    digest = runtime.project_function_source(project, _base_plan().base_transform)[1]
    seen = _merge_spies(monkeypatch)
    snapshot = {"source_versions": {"c.s.raw": 3}, "base_transform_sha256": digest}
    result = runtime.merge_feature_groups(None, _base_plan(), snapshot, {}, project)
    assert seen["base"] == ["raw-row", "derived"]
    assert result == {"version": 2}


def test_base_transform_edited_mid_run_is_rejected(tmp_path, monkeypatch):
    """A repair must not publish rows from code that differs from the initialized run."""
    project = _base_project(tmp_path)
    _merge_spies(monkeypatch)
    snapshot = {"source_versions": {"c.s.raw": 3}, "base_transform_sha256": "0" * 64}
    with pytest.raises(ValueError, match="base_transform changed"):
        runtime.merge_feature_groups(None, _base_plan(), snapshot, {}, project)


def test_initialize_pins_the_base_transform_code(tmp_path, monkeypatch):
    """The base code digest is fixed at initialize, like group transforms."""
    project = _base_project(tmp_path)
    monkeypatch.setattr(runtime, "current_version", lambda spark, table: 7)
    monkeypatch.setattr(runtime, "publication_state", lambda spark, table: None)
    snapshot = runtime.initialize_features(None, project, _base_plan(), "*")
    expected = runtime.project_function_source(project, _base_plan().base_transform)[1]
    assert snapshot["base_transform_sha256"] == expected
    assert snapshot["source_versions"] == {"c.s.raw": 7}


def test_base_only_feature_job_merges_after_initialize():
    """Without group tasks the merge must still wait for the pinned snapshot."""
    from skyulf.integrations.databricks.features.graph import build_feature_job

    training = {"tasks": [{"task_key": "t", "notebook_task": {}, "environment_key": "skyulf"}]}
    job = build_feature_job(_base_plan(), training)
    tasks = {task["task_key"]: task for task in job["tasks"]}
    assert set(tasks) == {"initialize_features", "merge_features"}
    assert tasks["merge_features"]["depends_on"] == [{"task_key": "initialize_features"}]
