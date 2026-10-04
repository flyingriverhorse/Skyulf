"""Model-set boundaries and generated reports preserve their declared contracts."""

import importlib
import json
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.model_set import ComponentReference, save_model_set
from skyulf.inference.model_set_scoring import score_model_set
from skyulf.integrations.databricks import model_set_batch as batch
from skyulf.integrations.databricks.admission import SingleWriterAdmission
from skyulf.integrations.databricks.monitoring_metrics import build_monitoring_report
from skyulf.integrations.databricks.monitoring_output import render_drift_output
from skyulf.integrations.mlflow.registry import ResolvedModel
from skyulf.pipeline import SkyulfPipeline

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Unexpected service connections must fail before these local repros leave the process."""

    def reject(*args, **kwargs):
        """Make accidental networking explicit instead of depending on sandbox denial."""
        raise AssertionError("Network is forbidden in the local platform repros")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def component(tmp_path, kind="float"):
    """Fit and save the declared input type without mocking model or artifact behavior."""
    values = list(range(12))
    data = pl.DataFrame({"x": [float(value) for value in values], "target": values})
    steps = []
    if kind.startswith("datetime") or kind == "date":
        dates = [datetime(2026, 1, 1) + timedelta(days=value) for value in values]
        if kind == "date":
            dates = [value.date() for value in dates]
        data = data.with_columns(pl.Series("special", dates))
        if kind.startswith("datetime"):
            dtype = {
                "datetime": pl.Datetime("us"),
                "datetime_ms": pl.Datetime("ms"),
                "datetime_ns": pl.Datetime("ns"),
                "datetime_utc": pl.Datetime("us", "UTC"),
                "datetime_vilnius": pl.Datetime("ns", "Europe/Vilnius"),
            }[kind]
            data = data.with_columns(pl.col("special").cast(dtype))
        steps = [
            {
                "name": "calendar",
                "transformer": "DateFeatures",
                "params": {
                    "columns": ["special"],
                    "features": ["day"],
                    "drop_original": True,
                },
            }
        ]
    elif kind == "categorical":
        data = data.with_columns(pl.Series("special", ["a", "b"] * 6, dtype=pl.Categorical))
        steps = [
            {
                "name": "category",
                "transformer": "OneHotEncoder",
                "params": {
                    "columns": ["special"],
                    "drop_original": True,
                    "handle_unknown": "ignore",
                },
            }
        ]
    pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": {"type": "linear_regression"}})
    pipeline.fit(data, target_column="target")
    path = tmp_path / "component"
    save_local_pipeline(pipeline, path)
    loaded = load_local_pipeline(path)
    reference = ComponentReference(
        name="workspace.test.component", version="1", digest=loaded.manifest.pipeline_sha256
    )
    return reference, path, data.drop("target")


def saved_set(tmp_path, kind="float"):
    """Create a complete real package and a concrete registry identity."""
    reference, path, query = component(tmp_path, kind)
    artifact = save_model_set(
        tmp_path / "set",
        {"amount": (reference, path)},
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    model = ResolvedModel(
        "workspace.test.set",
        "2",
        "models:/workspace.test.set/2",
        None,
        artifact.manifest.set_sha256,
    )
    return artifact, model, query


def test_d8_5_save_rejects_key_feature_overlap(tmp_path):
    """Unsupported join-key overlap must fail during packaging rather than first scoring."""
    pipeline = SkyulfPipeline({"modeling": {"type": "linear_regression"}})
    pipeline.fit(
        pd.DataFrame({"x": list(range(12)), "target": np.arange(12.0)}), target_column="target"
    )
    path = tmp_path / "component"
    save_local_pipeline(pipeline, path)
    digest = load_local_pipeline(path).manifest.pipeline_sha256
    reference = ComponentReference(name="workspace.test.component", version="1", digest=digest)
    with pytest.raises(ValueError, match="[Kk]ey|[Ii]nput|overlap"):
        save_model_set(
            tmp_path / "set",
            {"amount": (reference, path)},
            record_key_schema=(ColumnSpec(name="x", dtype="int64"),),
        )


def test_d8_5_disjoint_keys_still_score(tmp_path):
    """Valid packages retain key identity and the fitted numeric predictions."""
    artifact, _, query = saved_set(tmp_path)
    query = query.with_columns(pl.Series("id", list(range(12))))
    result = score_model_set(query, artifact)
    assert result.frame["id"].tolist() == list(range(12))
    np.testing.assert_allclose(
        result.frame["amount__prediction"].to_numpy(float), np.arange(12.0), atol=1e-10
    )


class Schema:
    """Expose the Spark schema interface used by public source/output admission."""

    def __init__(self, kinds):
        """Keep supported field types available by name and through schema.fields."""
        self.fields = [
            SimpleNamespace(name=name, dataType=SimpleNamespace(typeName=lambda value=kind: value))
            for name, kind in kinds.items()
        ]

    def __getitem__(self, name):
        """Resolve a named key field without bypassing admission validation."""
        return next(field for field in self.fields if field.name == name)


def empty_delta_transport(monkeypatch, artifact, model):
    """Replace remote I/O only; retain public admission, schema checks, plan and scoring."""
    source_columns = [column.name for column in artifact.manifest.input_schema]
    source_schema = {name: "long" if name == "id" else "string" for name in source_columns}
    target_schema = {name: kind for name, kind, _ in batch._table_columns(artifact)}
    frames = {
        "source": SimpleNamespace(columns=source_columns, schema=Schema(source_schema)),
        "target": SimpleNamespace(schema=Schema(target_schema)),
    }
    spark = SimpleNamespace(
        table=lambda name: frames[name], catalog=SimpleNamespace(tableExists=lambda name: True)
    )
    previous = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 0,
        "model_set_name": model.name,
        "model_set_version": "1",
        "model_set_digest": "older",
        "set_history": {},
    }
    empty = pd.DataFrame(columns=source_columns)
    monkeypatch.setattr(
        batch,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: (
                object() if name == "pyspark.sql.functions" else importlib.import_module(name)
            )
        ),
    )
    monkeypatch.setattr(batch, "table_identity", lambda spark, name: name)
    monkeypatch.setattr(batch, "require_incremental_change_feed", lambda *args: None)
    monkeypatch.setattr(
        batch,
        "latest_source_version",
        lambda spark, name: {
            "version": 0,
            "userMetadata": json.dumps(previous) if name == "target" else None,
        },
    )
    monkeypatch.setattr(batch, "check_incremental_bootstrap", lambda *args: None)
    monkeypatch.setattr(batch, "select_incremental_rows", lambda *args: empty)
    monkeypatch.setattr(batch, "bounded_frame", lambda *args: empty)
    monkeypatch.setattr(batch, "_output_frame", lambda spark, frame, *args: frame)
    commit = Mock(return_value=1)
    monkeypatch.setattr(batch, "_commit_set", commit)
    return spark, commit


@pytest.mark.parametrize(
    "kind",
    [
        "datetime",
        "datetime_ms",
        "datetime_ns",
        "datetime_utc",
        "datetime_vilnius",
        "date",
        "categorical",
        "float",
    ],
)
def test_d10_8_public_empty_full_rebuild_publishes_typed_empty_output(tmp_path, monkeypatch, kind):
    """An empty replacement clears old predictions for every accepted fitted input dtype."""
    artifact, model, _ = saved_set(tmp_path, kind)
    spark, commit = empty_delta_transport(monkeypatch, artifact, model)
    result = batch.run_model_set_batch(
        spark,
        model,
        artifact,
        source_table="source",
        prediction_table="target",
        admission=SingleWriterAdmission(),
        model_change_mode="full_rebuild",
        max_rows=20,
        max_bytes=1024 * 1024,
    )
    assert not result.noop and result.input_count == result.output_count == 0
    assert commit.call_count == 1 and commit.call_args.args[-1] == "overwrite"
    assert commit.call_args.args[1].empty
    assert result.manifest is not None
    assert result.manifest["model_set_digest"] == artifact.manifest.set_sha256


@pytest.mark.parametrize("kind", ["string", "boolean", "numeric"])
def test_d10_9_public_report_renders_measured_psi(kind):
    """Persisted categorical drift must be visible through the same report as numeric PSI."""
    low, high = {"string": ("a", "b"), "boolean": (False, True), "numeric": (0.0, 1.0)}[kind]
    reference = pl.DataFrame({"feature": [low] * 90 + [high] * 10})
    current = pl.DataFrame({"id": list(range(100)), "feature": [low] * 10 + [high] * 90})
    predictions = pl.DataFrame({"id": list(range(100)), "prediction": [0.0] * 100})
    report = build_monitoring_report(
        reference,
        current,
        predictions,
        None,
        feature_columns=("feature",),
        record_key_columns=("id",),
        target_column="target",
        result_available_at_column="available_at",
        as_of=datetime(2026, 10, 4, tzinfo=UTC),
        task="regression",
    )
    psi = next(metric for metric in report["metrics"] if metric["metric_name"].startswith("psi"))
    assert report["drifted_columns"] == 1 and psi["has_issue"]
    rendered = render_drift_output(
        {
            "report_json": json.dumps(report),
            "model_name": "workspace.test.model",
            "model_version": "1",
            "observed_at": "2026-10-04",
            "measured_at": "2026-10-04",
            "drifted_columns": 1,
        }
    )
    assert f"{psi['value']:.4g} / {psi['threshold']:.4g}" in rendered


def run_refresh(project):
    """Invoke the shipped operator script as users do, not its string-replacement helper."""
    result = subprocess.run(
        [sys.executable, str(project / "src/tools/refresh_training_graph.py")],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def refresh_project(project):
    """Keep the generated graph's exact refresh contract independent of optional CLI auth."""
    for directory in ("config", "resources", "src/modeling", "src/tools", "src/jobs"):
        (project / directory).mkdir(parents=True)
    (project / "config/workflow.json").write_text('{"training_layout": "multi_target"}')
    (project / "src/modeling/multi_model.py").write_text('MODELS = {"branch_1": {}}\n')
    for filename in (
        "src/tools/refresh_training_graph.py",
        "src/jobs/train_model.py",
        "src/jobs/shap_report.py",
    ):
        shutil.copyfile(TEMPLATE / filename, project / filename)
    (project / "resources/train.job.yml").write_text(
        """resources:
  jobs:
    train:
      tasks:
        - &branch_task
          task_key: initialize_run
          notebook_task:
            notebook_path: ../src/jobs/initialize_models.py
            base_parameters: &branch_parameters
              model_keys_json: '["branch_1"]'
        # BEGIN MODEL TASKS
        - <<: *branch_task
          task_key: train_branch_1
          depends_on:
            - task_key: initialize_run
          notebook_task:
            notebook_path: ../src/jobs/train_model.py
            base_parameters:
              <<: *branch_parameters
              model_key: "branch_1"
        - <<: *branch_task
          task_key: shap_branch_1
          depends_on:
            - task_key: train_branch_1
          notebook_task:
            notebook_path: ../src/jobs/shap_report.py
            base_parameters:
              <<: *branch_parameters
              reference_json: '{{tasks.train_branch_1.values.reference_json}}'
        # END MODEL TASKS
        - task_key: register_model_set
          depends_on:
            # BEGIN MODEL DEPENDENCIES
            - task_key: train_branch_1
            # END MODEL DEPENDENCIES
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize("old,new", [("re", "a"), ("model", "churn"), ("revenue", "turnover")])
@pytest.mark.parametrize("check", ["paths", "idempotence"])
def test_d11_3_public_refresh_preserves_notebooks_and_repeated_output(tmp_path, old, new, check):
    """Legitimate model identifiers cannot rewrite notebook filenames or lose SHAP tasks on retry."""
    project = tmp_path / "project"
    refresh_project(project)
    declarations = project / "src/modeling/multi_model.py"
    source = declarations.read_text(encoding="utf-8")
    declarations.write_text(
        source + f"\nMODELS = {{{old!r}: next(iter(MODELS.values()))}}\n", encoding="utf-8"
    )
    run_refresh(project)
    declarations.write_text(
        declarations.read_text() + f"\nMODELS = {{{new!r}: next(iter(MODELS.values()))}}\n",
        encoding="utf-8",
    )
    run_refresh(project)
    graph = project / "resources/train.job.yml"
    before = graph.read_text(encoding="utf-8")
    if check == "paths":
        tasks = {
            task["task_key"]: task
            for task in yaml.safe_load(before)["resources"]["jobs"]["train"]["tasks"]
        }
        assert (
            tasks[f"train_{new}"]["notebook_task"]["notebook_path"] == "../src/jobs/train_model.py"
        )
        assert (
            tasks[f"shap_{new}"]["notebook_task"]["notebook_path"] == "../src/jobs/shap_report.py"
        )
        assert tasks[f"shap_{new}"]["depends_on"] == [{"task_key": f"train_{new}"}]
        assert tasks[f"shap_{new}"]["notebook_task"]["base_parameters"]["reference_json"] == (
            "{{tasks.train_" + new + ".values.reference_json}}"
        )
        assert all(
            (project / "resources" / tasks[name]["notebook_task"]["notebook_path"]).is_file()
            for name in (f"train_{new}", f"shap_{new}")
        )
    else:
        run_refresh(project)
        assert graph.read_text(encoding="utf-8") == before
