"""Offline notebook transport checks; these do not exercise Databricks or Spark."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.features.config import (
    FeatureGroup,
    FeaturePlan,
    validate_feature_plan,
)
from skyulf.integrations.databricks.features.graph import plan_digest
from skyulf.integrations.databricks.features.runtime import (
    build_feature_group,
    initialize_features,
    merge_feature_groups,
)
from skyulf.integrations.databricks.jobs.features import feature_task


@dataclass
class _Widgets:
    """Expose the widget read boundary and make unexpected reads observable."""

    values: dict[str, str]
    reads: list[str] = field(default_factory=list)

    def get(self, name: str) -> str:
        """Use Databricks-style required values without inventing missing defaults."""
        self.reads.append(name)
        return self.values[name]


@dataclass
class _TaskValues:
    """Record the notebook's published JSON envelope without remote state."""

    writes: list[tuple[str, str]] = field(default_factory=list)

    def set(self, *, key: str, value: str) -> None:
        """Retain exact serialized values to detect transport contract changes."""
        self.writes.append((key, value))


class _NoSpark:
    """Fail if a preflight-only test touches a Spark operation."""

    def __getattr__(self, name: str):
        """Keep cloud and local JVM behavior outside these validation probes."""
        raise AssertionError(f"Spark was accessed before validation: {name}")


def _plan() -> FeaturePlan:
    """Declare two independent domains so swapped receipt routing is detectable."""
    return FeaturePlan(
        base_table="catalog.demo.observations",
        output_table="catalog.demo.training_features",
        keys=("customer_id",),
        timestamp="event_time",
        groups=(
            FeatureGroup(
                "company",
                "catalog.demo.raw_company",
                "catalog.demo.company_features",
                "src/features/groups/company.py:compute",
                ("size",),
            ),
            FeatureGroup(
                "activity",
                "catalog.demo.raw_activity",
                "catalog.demo.activity_features",
                "src/features/groups/activity.py:compute",
                ("spend",),
                lookup="asof",
            ),
        ),
    )


@pytest.fixture
def notebook(tmp_path):
    """Read a real YAML file through the production notebook configuration path."""
    path = tmp_path / "config/features.yml"
    path.parent.mkdir()
    path.write_text(
        "version: 1\n"
        "base_table: catalog.demo.observations\n"
        "output_table: catalog.demo.training_features\n"
        "keys: [customer_id]\ntimestamp: event_time\n"
        "groups:\n"
        "  company:\n"
        "    source_table: catalog.demo.raw_company\n"
        "    output_table: catalog.demo.company_features\n"
        "    transform: src/features/groups/company.py:compute\n"
        "    columns: [size]\n"
        "  activity:\n"
        "    source_table: catalog.demo.raw_activity\n"
        "    output_table: catalog.demo.activity_features\n"
        "    transform: src/features/groups/activity.py:compute\n"
        "    columns: [spend]\n    lookup: asof\n",
        encoding="utf-8",
    )
    widgets = _Widgets(
        {
            "config_path": str(path),
            "plan_sha256": plan_digest(_plan()),
            "phase": "initialize",
            "selected_groups": "company",
        }
    )
    task_values = _TaskValues()
    dbutils = SimpleNamespace(widgets=widgets, jobs=SimpleNamespace(taskValues=task_values))
    return SimpleNamespace(path=path, dbutils=dbutils, widgets=widgets, task_values=task_values)


@pytest.fixture
def operations(monkeypatch):
    """Replace only the three external data-operation boundaries."""
    initialize = Mock(return_value={"selected": ["company"], "source_versions": {"raw": 7}})
    build = Mock(return_value={"table": "catalog.demo.company_features", "version": 8})
    merge = Mock(return_value={"table": "catalog.demo.training_features", "version": 9})
    monkeypatch.setattr(feature_task, "initialize_features", initialize)
    monkeypatch.setattr(feature_task, "build_feature_group", build)
    monkeypatch.setattr(feature_task, "merge_feature_groups", merge)
    return SimpleNamespace(initialize=initialize, build=build, merge=merge)


def _assert_no_operations(operations) -> None:
    """Ensure failed preflight cannot contact any data-operation boundary."""
    operations.initialize.assert_not_called()
    operations.build.assert_not_called()
    operations.merge.assert_not_called()


def test_initialize_routes_selected_groups_and_publishes_snapshot(notebook, operations, capsys):
    """The initializer must receive the project root and publish the snapshot widget key."""
    spark = _NoSpark()
    result = feature_task.run_feature_notebook(spark, notebook.dbutils)
    operations.initialize.assert_called_once_with(
        spark, notebook.path.parent.parent, _plan(), "company"
    )
    operations.build.assert_not_called()
    operations.merge.assert_not_called()
    assert result == {"selected": ["company"], "source_versions": {"raw": 7}}
    assert notebook.task_values.writes == [
        ("snapshot_json", '{"selected": ["company"], "source_versions": {"raw": 7}}')
    ]
    assert json.loads(capsys.readouterr().out) == {
        "phase": "initialize",
        "selected": ["company"],
        "source_versions": {"raw": 7},
    }


@pytest.mark.parametrize("phase", ["initialize", "group", "merge"])
def test_stale_graph_rejected_before_any_data_action(notebook, operations, phase):
    """Editing feature settings without regenerating the deployed graph must stop all phases."""
    notebook.widgets.values.update(phase=phase, plan_sha256="old-deployed-digest")
    with pytest.raises(ValueError, match="stale"):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    _assert_no_operations(operations)
    assert notebook.task_values.writes == []
    assert notebook.widgets.reads == ["config_path", "plan_sha256"]


def test_disabled_config_needs_no_graph_or_data_widgets(notebook, operations):
    """An obsolete deployed feature job must stop cleanly after features are disabled."""
    notebook.path.write_text("version: 1\ngroups: {}\n", encoding="utf-8")
    notebook.widgets.values = {"config_path": str(notebook.path)}
    with pytest.raises(ValueError, match="disabled"):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    _assert_no_operations(operations)
    assert notebook.task_values.writes == []


def test_unknown_phase_rejected_before_data_action(notebook, operations):
    """A mistyped task phase must never fall through to a write-producing operation."""
    notebook.widgets.values["phase"] = "rebuild_everything"
    with pytest.raises(ValueError, match="Unknown feature task phase"):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    _assert_no_operations(operations)
    assert notebook.task_values.writes == []


def test_group_routes_decoded_snapshot_and_group_name(notebook, operations):
    """Group repair must consume the initializer's exact snapshot and selected domain."""
    notebook.widgets.values.update(
        phase="group",
        group="company",
        snapshot_json='{"selected": ["company"], "source_versions": {"raw": 7}}',
    )
    spark = _NoSpark()
    result = feature_task.run_feature_notebook(spark, notebook.dbutils)
    operations.build.assert_called_once_with(
        spark,
        notebook.path.parent.parent,
        _plan(),
        {"selected": ["company"], "source_versions": {"raw": 7}},
        "company",
    )
    operations.initialize.assert_not_called()
    operations.merge.assert_not_called()
    assert result == {"table": "catalog.demo.company_features", "version": 8}
    assert notebook.task_values.writes == [
        ("receipt_json", '{"table": "catalog.demo.company_features", "version": 8}')
    ]


def test_merge_routes_named_receipts_without_swapping_groups(notebook, operations):
    """Fan-in must map each task's receipt to its configured domain rather than widget order."""
    notebook.widgets.values.update(
        phase="merge",
        snapshot_json='{"source_versions": {"base": 5}}',
        receipt_activity_json='{"table": "catalog.demo.activity_features", "version": 13}',
        receipt_company_json='{"table": "catalog.demo.company_features", "version": 8}',
    )
    spark = _NoSpark()
    result = feature_task.run_feature_notebook(spark, notebook.dbutils)
    operations.merge.assert_called_once_with(
        spark,
        _plan(),
        {"source_versions": {"base": 5}},
        {
            "company": {"table": "catalog.demo.company_features", "version": 8},
            "activity": {"table": "catalog.demo.activity_features", "version": 13},
        },
    )
    operations.initialize.assert_not_called()
    operations.build.assert_not_called()
    assert result == {"table": "catalog.demo.training_features", "version": 9}
    assert notebook.task_values.writes == [
        ("receipt_json", '{"table": "catalog.demo.training_features", "version": 9}')
    ]


@pytest.mark.parametrize("phase", ["group", "merge"])
def test_malformed_receipt_json_stops_before_data_action(notebook, operations, phase):
    """Truncated cross-task evidence must not reach a writer or publish another receipt."""
    notebook.widgets.values.update(
        phase=phase,
        group="company",
        snapshot_json="{",
        receipt_company_json="{",
        receipt_activity_json="{}",
    )
    with pytest.raises(json.JSONDecodeError):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    _assert_no_operations(operations)
    assert notebook.task_values.writes == []


@pytest.mark.parametrize(
    "phase,field",
    [("group", "snapshot_json"), ("merge", "receipt_company_json")],
)
def test_nonfinite_incoming_json_stops_before_data_action(notebook, operations, phase, field):
    """Nonstandard NaN task evidence must not reach feature production or merge."""
    notebook.widgets.values.update(
        phase=phase,
        group="company",
        snapshot_json="{}",
        receipt_company_json="{}",
        receipt_activity_json="{}",
    )
    notebook.widgets.values[field] = '{"version": NaN}'
    with pytest.raises(ValueError):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    _assert_no_operations(operations)
    assert notebook.task_values.writes == []


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_task_value_obeys_exact_48_kib_serialized_boundary(notebook, operations, extra_bytes):
    """Task evidence at the limit is allowed; one extra serialized byte must not be sent."""
    # {"evidence": ""} is 16 ASCII bytes with Python's default JSON separators.
    operations.initialize.return_value = {"evidence": "x" * (48 * 1024 - 16 + extra_bytes)}
    if extra_bytes:
        with pytest.raises(ValueError, match="task-value limit"):
            feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
        assert notebook.task_values.writes == []
    else:
        result = feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
        key, serialized = notebook.task_values.writes[0]
        assert result["evidence"].startswith("x")
        assert key == "snapshot_json"
        assert len(serialized.encode("utf-8")) == 48 * 1024


def test_task_value_limit_uses_serialized_unicode_size(notebook, operations):
    """Non-ASCII metadata must not bypass the limit through a short Python string length."""
    operations.initialize.return_value = {"evidence": "\u0131" * 9000}
    with pytest.raises(ValueError, match="task-value limit"):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    assert notebook.task_values.writes == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_evidence_is_never_published(notebook, operations, value):
    """NaN and infinity cannot enter a repair receipt that other tasks must parse."""
    operations.initialize.return_value = {"metadata": {"invalid": value}}
    with pytest.raises(ValueError, match="JSON"):
        feature_task.run_feature_notebook(_NoSpark(), notebook.dbutils)
    assert notebook.task_values.writes == []


@pytest.mark.parametrize(
    "invalid_plan,reason",
    [
        (replace(_plan(), output_table="observations"), "catalog.schema.table"),
        (replace(_plan(), output_table="CATALOG.DEMO.OBSERVATIONS"), "separate"),
        (replace(_plan(), keys=("customer_id", "CUSTOMER_ID")), "distinct"),
        (replace(_plan(), keys="customer_id"), "keys"),
        (
            replace(_plan(), groups=(replace(_plan().groups[0], columns="size"),)),
            "columns",
        ),
        (replace(_plan(), timestamp="customer_id"), "separate"),
        (replace(_plan(), groups=(_plan().groups[0], _plan().groups[0])), "distinct"),
        (
            replace(_plan(), groups=(replace(_plan().groups[0], transform="../outside.py:run"),)),
            "transform",
        ),
        (replace(_plan(), groups=(replace(_plan().groups[0], allow_missing="false"),)), "boolean"),
        (
            replace(_plan(), groups=(replace(_plan().groups[0], columns=("customer_id",)),)),
            "overlap",
        ),
    ],
)
def test_direct_dataclass_plans_cannot_bypass_validation_before_spark(
    tmp_path, invalid_plan, reason
):
    """Public runtime helpers must enforce YAML-equivalent ownership and grain checks."""
    with pytest.raises(ValueError, match=reason):
        validate_feature_plan(invalid_plan)
    spark = _NoSpark()
    with pytest.raises(ValueError, match=reason):
        initialize_features(spark, Path(tmp_path), invalid_plan, "*")
    with pytest.raises(ValueError, match=reason):
        build_feature_group(spark, Path(tmp_path), invalid_plan, {}, "company")
    with pytest.raises(ValueError, match=reason):
        merge_feature_groups(spark, invalid_plan, {}, {})
