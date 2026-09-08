"""Selected-node preview receipts preserve measured data and execution identity."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._internal._routers import preview as preview_mod
from skyulf.data.dataset import SplitDataset


def _node(node_id, step_type="StandardScaler", inputs=None, params=None):
    """Keep fixtures explicit about the actual graph sent to the endpoint."""
    return {
        "node_id": node_id,
        "step_type": step_type,
        "inputs": inputs or [],
        "params": params or {},
    }


def _source(node_id="source"):
    """Use a catalog path so endpoint fixtures need no persisted dataset row."""
    return _node(node_id, "data_loader", params={"path": node_id})


@pytest.fixture
def preview_client(monkeypatch, tmp_path):
    """Run the real preview router and engine with an in-memory external catalog."""
    catalog = MagicMock()
    catalog.load.return_value = pd.DataFrame({"value": [1.0, 3.0]})
    sync_session = MagicMock()
    directories = []

    def make_temp_dir(**kwargs):
        """Expose request-owned paths so failure cleanup is observable."""
        directory = tmp_path / str(uuid4())
        directory.mkdir()
        directories.append(directory)
        return str(directory)

    def request_session():
        """Supply the dependency without opening a database connection."""
        return MagicMock()

    monkeypatch.setattr(preview_mod, "tempfile", SimpleNamespace(mkdtemp=make_temp_dir))
    monkeypatch.setattr(preview_mod, "resolve_pipeline_nodes", AsyncMock(return_value={}))
    monkeypatch.setattr(preview_mod, "create_catalog_from_options", lambda *a, **k: catalog)
    monkeypatch.setattr(preview_mod.db_engine, "sync_session_factory", lambda: sync_session)
    monkeypatch.setattr(preview_mod, "_generate_recommendations", lambda data: [])
    app = FastAPI()
    app.include_router(preview_mod.router, prefix="/pipeline")
    app.dependency_overrides[preview_mod.get_async_session] = request_session
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, catalog=catalog, directories=directories, session=sync_session
        )


def _preview(harness, nodes, selected="scale", *, inspect_all=False, pipeline_id="inspection"):
    """Submit the existing request body with optional selected-node or full capture."""
    params = {"inspect_node_id": selected} if selected is not None else {}
    if inspect_all:
        params["inspect_all"] = "true"
    response = harness.client.post(
        "/pipeline/preview",
        params=params,
        json={"pipeline_id": pipeline_id, "nodes": nodes},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_inspection_returns_actual_before_and_after(preview_client):
    """The selected node must expose its own transform rather than terminal data."""
    response = _preview(
        preview_client,
        [_source(), _node("scale", inputs=["source"]), _node("end", inputs=["scale"])],
    )
    entry = response["node_inspections"][0]
    assert UUID(response["run_id"])
    assert entry["node_id"] == "scale"
    assert entry["input"]["status"] == entry["output"]["status"] == "available"
    assert entry["input"]["tables"][0]["rows"] == [{"value": 1.0}, {"value": 3.0}]
    assert entry["output"]["tables"][0]["rows"] == [{"value": -1.0}, {"value": 1.0}]
    assert all(not path.exists() for path in preview_client.directories)


def test_inspection_snapshots_input_before_in_place_mutation(preview_client, monkeypatch):
    """A transformer modifying its input must not rewrite the captured before sample."""

    def mutate(self, node, job_id="unknown"):
        """Exercise the real resolver with a deliberately in-place transformer."""
        data = self._get_input(node)
        data.loc[:, "value"] += 10
        self.artifact_store.save(node.node_id, data)
        return node.node_id, {}

    monkeypatch.setattr(PipelineEngine, "_run_transformer", mutate)
    response = _preview(preview_client, [_source(), _node("scale", inputs=["source"])])
    entry = response["node_inspections"][0]
    assert entry["input"]["tables"][0]["rows"][0] == {"value": 1.0}
    assert entry["output"]["tables"][0]["rows"][0] == {"value": 11.0}


def test_inspection_captures_merged_input(preview_client):
    """Fan-in inspection must measure the resolved union, preserving both parents."""
    preview_client.catalog.load.side_effect = [
        pd.DataFrame({"left": [1.0, 3.0]}),
        pd.DataFrame({"right": [2.0, 4.0]}),
    ]
    response = _preview(
        preview_client,
        [_source("left"), _source("right"), _node("scale", inputs=["left", "right"])],
    )
    table = response["node_inspections"][0]["input"]["tables"][0]
    assert table["port"] == "input"
    assert table["column_count"] == 2
    assert table["rows"] == [{"left": 1.0, "right": 2.0}, {"left": 3.0, "right": 4.0}]


def test_source_inspection_has_explicit_missing_input(preview_client):
    """A source has measured output but no upstream input to invent."""
    response = _preview(preview_client, [_source()], selected="source")
    entry = response["node_inspections"][0]
    assert entry["input"]["status"] == "unavailable"
    assert entry["input"]["reason"]
    assert entry["input"]["tables"] == []
    assert entry["output"]["tables"][0]["row_count"] == 2


@pytest.mark.parametrize("selected", ["broken", "later"])
def test_failed_and_unexecuted_nodes_have_explicit_states(preview_client, selected):
    """Failure must preserve resolved input and distinguish nodes never reached."""
    response = _preview(
        preview_client,
        [
            _source(),
            _node("broken", "unknown-transformer", ["source"]),
            _node("later", inputs=["broken"]),
        ],
        selected=selected,
    )
    entry = response["node_inspections"][0]
    assert response["status"] == "failed"
    assert entry["output"]["status"] == ("error" if selected == "broken" else "unavailable")
    assert entry["output"]["reason"]
    assert entry["output"]["tables"] == []
    assert entry["input"]["status"] == ("available" if selected == "broken" else "unavailable")


@pytest.mark.parametrize("selected,kind", [("model", "training"), ("preview", "data_preview")])
def test_skipped_preview_nodes_are_unavailable(preview_client, selected, kind):
    """Inspection cannot pretend excluded training or background preview nodes executed."""
    response = _preview(
        preview_client, [_source(), _node(selected, kind, ["source"])], selected=selected
    )
    entries = response["node_inspections"]
    assert entries
    assert all(entry["input"]["status"] == "unavailable" for entry in entries)
    assert all(entry["output"]["status"] == "unavailable" for entry in entries)
    assert all(entry["output"]["reason"] for entry in entries)


def test_selected_training_node_explains_why_preview_skipped_it(preview_client):
    """A selected model must explain preview's training exclusion rather than imply an unknown run."""
    response = _preview(
        preview_client, [_source(), _node("model", "training", ["source"])], selected="model"
    )
    entry = response["node_inspections"][0]
    assert entry["input"]["status"] == entry["output"]["status"] == "unavailable"
    assert "Training nodes are skipped" in entry["input"]["reason"]
    assert "Training nodes are skipped" in entry["output"]["reason"]


def test_shared_node_executions_keep_distinct_branch_snapshots(preview_client, monkeypatch):
    """Shared artifact keys and repeated labels must not collapse branch executions."""
    preview_client.catalog.load.side_effect = [
        pd.DataFrame({"value": [1.0, 3.0]}),
        pd.DataFrame({"value": [10.0, 30.0]}),
    ]
    monkeypatch.setattr(preview_mod, "_branch_label", lambda *args: "Same label")
    response = _preview(
        preview_client,
        [_source(), _node("a", inputs=["source"]), _node("b", inputs=["source"])],
        selected="source",
    )
    first, second = response["node_inspections"]
    assert first["branch_id"] != second["branch_id"]
    assert first["branch_label"] == second["branch_label"] == "Same label"
    assert first["output"]["tables"][0]["rows"][0] == {"value": 1.0}
    assert second["output"]["tables"][0]["rows"][0] == {"value": 10.0}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_samples_are_bounded_with_actual_shape(preview_client, engine):
    """Wide or long data must retain true preview counts while bounding the receipt."""
    data = {f"column_{i}": ["x" * 1000] * 60 for i in range(105)}
    preview_client.catalog.load.return_value = (
        pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    )
    response = _preview(preview_client, [_source()], selected="source")
    table = response["node_inspections"][0]["output"]["tables"][0]
    assert (table["row_count"], table["column_count"]) == (60, 105)
    assert len(table["columns"]) == len(table["rows"][0]) == 100
    assert 0 < len(table["rows"]) <= 50
    assert len(table["rows"][0]["column_0"]) <= 500
    assert table["truncated"] is True


def test_empty_frame_retains_schema(preview_client):
    """Empty measured output is available data with dtypes, not a missing result."""
    preview_client.catalog.load.return_value = pd.DataFrame({"value": pd.Series(dtype="Int64")})
    response = _preview(preview_client, [_source()], selected="source")
    side = response["node_inspections"][0]["output"]
    assert side["status"] == "available"
    assert side["tables"][0]["columns"] == [{"name": "value", "dtype": "Int64"}]
    assert side["tables"][0]["rows"] == []
    assert side["tables"][0]["row_count"] == 0


@pytest.mark.parametrize("as_dict", [False, True])
def test_split_xy_tables_keep_port_and_split_identity(preview_client, monkeypatch, as_dict):
    """Train/test/validation feature and target samples must remain six separate tables."""
    parts = {
        "train": (pd.DataFrame({"value": [1, 2]}), pd.Series([0, 1], name="target")),
        "test": (pl.DataFrame({"value": [3]}), pl.Series("target", [1])),
        "validation": (pd.DataFrame({"value": [4]}), np.array([0])),
    }

    def produce_split(self, node, job_id="unknown"):
        """Supply supported engine artifacts without coupling this contract to a splitter."""
        data = parts if as_dict else SplitDataset(**parts)
        self.artifact_store.save(node.node_id, data)
        return node.node_id

    monkeypatch.setattr(PipelineEngine, "_run_data_loader", produce_split)
    response = _preview(preview_client, [_source()], selected="source")
    tables = response["node_inspections"][0]["output"]["tables"]
    assert [(t["split"], t["port"], t["row_count"]) for t in tables] == [
        ("train", "X", 2),
        ("train", "y", 2),
        ("test", "X", 1),
        ("test", "y", 1),
        ("validation", "X", 1),
        ("validation", "y", 1),
    ]
    assert tables[3]["rows"] == [{"target": 1}]


def test_unknown_selection_is_explicitly_unavailable(preview_client):
    """A missing requested node still produces an honest receipt instead of no explanation."""
    response = _preview(preview_client, [_source()], selected="missing")
    entry = response["node_inspections"][0]
    assert entry["node_id"] == "missing"
    assert entry["output"]["status"] == "unavailable"
    assert entry["output"]["reason"]


def test_default_preview_preserves_terminal_payload_without_inspection(preview_client):
    """Callers that omit inspection retain their terminal sample and no captured nodes."""
    response = _preview(preview_client, [_source()], selected=None)
    assert response["preview_data"] == [{"value": 1.0}, {"value": 3.0}]
    assert response["node_inspections"] == []
    assert response["run_id"] is None


def test_resolution_failure_cleans_up_temporary_store(preview_client, monkeypatch):
    """Dataset lookup errors must release the temporary directory even before execution."""
    monkeypatch.setattr(
        preview_mod,
        "resolve_pipeline_nodes",
        AsyncMock(side_effect=HTTPException(status_code=404, detail="Dataset missing")),
    )
    response = preview_client.client.post(
        "/pipeline/preview", json={"pipeline_id": "missing", "nodes": [_source()]}
    )
    assert response.status_code == 404
    assert preview_client.directories
    assert all(not path.exists() for path in preview_client.directories)


def test_cell_values_are_json_safe_detached_and_bounded():
    """Nested cells, nulls and timestamps must serialize without leaking mutable references."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    nested = ["x" * 1000] * 100
    frame = pd.DataFrame(
        {
            "value": [np.nan, np.inf, pd.NA, pd.Timestamp("2026-09-08"), nested],
        }
    )
    side = snapshot_side(frame, "output")
    nested.append("later")
    rows = side["tables"][0]["rows"]
    assert rows[:3] == [{"value": None}, {"value": None}, {"value": None}]
    assert rows[3]["value"].startswith("2026-09-08")
    assert isinstance(rows[4]["value"], str) and len(rows[4]["value"]) <= 500
    assert "later" not in json.dumps(side, allow_nan=False)


def test_unsupported_artifact_is_unavailable():
    """Models and arbitrary artifacts must never masquerade as an empty measured frame."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    side = snapshot_side({"model": object()}, "output")
    assert side["status"] == "unavailable"
    assert side["reason"]
    assert side["tables"] == []


def test_total_sample_budget_preserves_all_split_schemas():
    """Six wide tables must share a bounded sample budget without losing split schemas."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    frame = pd.DataFrame({f"column_{i}": ["x" * 1000] * 100 for i in range(100)})
    side = snapshot_side(SplitDataset((frame, frame), (frame, frame), (frame, frame)), "output")
    tables = side["tables"]
    samples = [row for table in tables for row in table["rows"]]
    assert len(tables) == 6
    assert all(len(table["columns"]) == 100 and table["row_count"] == 100 for table in tables)
    assert all(table["rows"] and table["truncated"] for table in tables)
    assert len(json.dumps(samples, separators=(",", ":"))) <= 256 * 1024


def test_column_named_to_native_is_still_a_measured_frame():
    """Pandas column attribute lookup must not mistake ordinary data for an engine wrapper."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    side = snapshot_side(pd.DataFrame({"to_native": [3]}), "input")
    assert side["status"] == "available"
    assert side["tables"][0]["rows"] == [{"to_native": 3}]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_decimal_samples_preserve_exact_values_and_scale(engine):
    """Decimal measurements must retain precision and trailing zeros without false truncation."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    values = [Decimal("1.25"), Decimal("2.50"), Decimal("12345678901234567890.12")]
    frame = (
        pd.DataFrame({"value": values}) if engine == "pandas" else pl.DataFrame({"value": values})
    )
    side = snapshot_side(frame, "output")
    assert side["status"] == "available"
    assert side["tables"][0]["rows"] == [
        {"value": "1.25"},
        {"value": "2.50"},
        {"value": "12345678901234567890.12"},
    ]
    assert side["tables"][0]["truncated"] is False


def test_large_decimal_samples_obey_display_bounds():
    """Exact Decimal rendering must still respect the maximum displayed cell length."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    side = snapshot_side(pd.DataFrame({"value": [Decimal("1" * 600 + ".25")]}), "output")
    table = side["tables"][0]
    value = table["rows"][0]["value"]
    assert value.startswith("11111")
    assert len(value) <= 500
    assert table["truncated"] is True


def test_default_preview_performs_no_sample_capture(preview_client, monkeypatch):
    """Adding inspection must not add serialization work to requests that do not ask for it."""
    import backend.ml_pipeline._execution.engine as engine_mod

    capture = MagicMock(side_effect=AssertionError("Unexpected sample capture"))
    monkeypatch.setattr(engine_mod, "snapshot_side", capture)
    response = _preview(preview_client, [_source(), _node("scale", inputs=["source"])], None)
    assert response["status"] == "success"
    capture.assert_not_called()


def test_input_resolution_failure_has_error_receipt(preview_client):
    """A failed artifact lookup must distinguish unresolved input from an empty table."""
    response = _preview(preview_client, [_node("scale", inputs=["missing"])])
    entry = response["node_inspections"][0]
    assert entry["input"]["status"] == entry["output"]["status"] == "error"
    assert entry["input"]["reason"]
    assert entry["input"]["tables"] == []


def test_sample_capture_failure_preserves_execution_and_cleans_store(preview_client, monkeypatch):
    """Inspection serialization errors are advisory and must still release temporary artifacts."""
    from backend.ml_pipeline._execution.engine import _inspection

    monkeypatch.setattr(
        _inspection, "_snapshot_table", MagicMock(side_effect=ValueError("Bad sample"))
    )
    response = _preview(preview_client, [_source()], "source")
    entry = response["node_inspections"][0]
    assert response["status"] == "success"
    assert entry["output"]["status"] == "error"
    assert all(not path.exists() for path in preview_client.directories)


def test_bulk_inspection_captures_all_nodes_in_one_execution(preview_client, monkeypatch):
    """Changing selected nodes must reuse one receipt without reloading or rerunning the graph."""
    executions = []
    execute = PipelineEngine._execute_node

    def record_execution(self, node, job_id="unknown"):
        """Count actual node dispatch while preserving the entire engine execution path."""
        executions.append(node.node_id)
        return execute(self, node, job_id)

    monkeypatch.setattr(PipelineEngine, "_execute_node", record_execution)
    response = _preview(
        preview_client,
        [_source(), _node("scale", inputs=["source"]), _node("end", inputs=["scale"])],
        selected=None,
        inspect_all=True,
    )
    entries = {entry["node_id"]: entry for entry in response["node_inspections"]}
    assert UUID(response["run_id"])
    assert set(entries) == {"source", "scale", "end"}
    assert entries["source"]["input"]["status"] == "unavailable"
    assert entries["source"]["output"]["tables"][0]["rows"] == [{"value": 1.0}, {"value": 3.0}]
    assert entries["scale"]["input"]["tables"][0]["rows"] == [{"value": 1.0}, {"value": 3.0}]
    assert entries["scale"]["output"]["tables"][0]["rows"] == [{"value": -1.0}, {"value": 1.0}]
    assert entries["end"]["input"]["tables"][0]["rows"] == [{"value": -1.0}, {"value": 1.0}]
    assert executions == ["source", "scale", "end"]
    preview_client.catalog.load.assert_called_once_with("source", limit=1000)


def test_bulk_inspection_keeps_each_shared_node_branch_execution(preview_client, monkeypatch):
    """Repeated shared nodes need distinct snapshots while preserving existing branch work."""
    preview_client.catalog.load.side_effect = [
        pd.DataFrame({"value": [1.0, 3.0]}),
        pd.DataFrame({"value": [10.0, 30.0]}),
    ]
    monkeypatch.setattr(preview_mod, "_branch_label", lambda *args: "Same label")
    response = _preview(
        preview_client,
        [_source(), _node("a", inputs=["source"]), _node("b", inputs=["source"])],
        selected="a",
        inspect_all=True,
    )
    entries = response["node_inspections"]
    sources = [entry for entry in entries if entry["node_id"] == "source"]
    assert len(entries) == 4
    assert len(sources) == 2
    assert sources[0]["branch_id"] != sources[1]["branch_id"]
    assert sources[0]["branch_label"] == sources[1]["branch_label"] == "Same label"
    assert sources[0]["output"]["tables"][0]["rows"][0] == {"value": 1.0}
    assert sources[1]["output"]["tables"][0]["rows"][0] == {"value": 10.0}
    assert preview_client.catalog.load.call_count == 2


def test_bulk_inspection_includes_failed_unexecuted_and_skipped_nodes(preview_client):
    """A failed branch must still explain all configured nodes, including excluded terminals."""
    response = _preview(
        preview_client,
        [
            _source(),
            _node("broken", "unknown-transformer", ["source"]),
            _node("later", inputs=["broken"]),
            _node("model", "training", ["later"]),
            _node("preview", "data_preview", ["source"]),
        ],
        selected=None,
        inspect_all=True,
    )
    entries = {entry["node_id"]: entry for entry in response["node_inspections"]}
    assert set(entries) == {"source", "broken", "later", "model", "preview"}
    assert response["status"] == "failed"
    assert entries["source"]["output"]["status"] == "available"
    assert entries["broken"]["input"]["status"] == "available"
    assert entries["broken"]["output"]["status"] == "error"
    for node_id in ("later", "model", "preview"):
        assert entries[node_id]["output"]["status"] == "unavailable"
        assert entries[node_id]["output"]["reason"]


@pytest.mark.parametrize("suffix", ["", "x" * 700], ids=["short_error", "long_error"])
def test_bulk_failure_receipt_preserves_bounded_error_in_its_branch(preview_client, suffix):
    """An inspector-only refresh must reveal the real error without tainting successful branches."""
    response = _preview(
        preview_client,
        [
            _source(),
            _node("broken", "unknown-transformer" + suffix, ["source"]),
            _node("later", inputs=["broken"]),
            _node("healthy", inputs=["source"]),
        ],
        selected=None,
        inspect_all=True,
    )
    entries = response["node_inspections"]
    broken = next(entry for entry in entries if entry["node_id"] == "broken")
    healthy = next(entry for entry in entries if entry["node_id"] == "healthy")
    later = next(entry for entry in entries if entry["node_id"] == "later")
    assert response["status"] == "failed"
    assert broken["output"]["status"] == "error"
    assert "Unknown step type: unknown-transformer" in broken["output"]["reason"]
    assert len(broken["output"]["reason"]) <= 500
    assert broken["input"]["tables"][0]["rows"] == [{"value": 1.0}, {"value": 3.0}]
    assert healthy["branch_id"] != broken["branch_id"]
    assert healthy["output"]["status"] == "available"
    assert healthy["output"]["reason"] is None
    assert later["output"]["status"] == "unavailable"
    assert later["output"]["tables"] == []
    assert all(
        entry["output"]["tables"][0]["rows"] == [{"value": 1.0}, {"value": 3.0}]
        for entry in entries
        if entry["node_id"] == "source"
    )


def test_bulk_samples_share_global_budget_without_losing_later_schemas(preview_client, monkeypatch):
    """Large graphs must share the sample budget fairly while retaining every measured shape."""
    preview_client.catalog.load.return_value = pd.DataFrame(
        {f"column_{i}": ["x" * 1000] * 100 for i in range(100)}
    )

    def passthrough(self, node, job_id="unknown"):
        """Exercise actual resolution and artifact writes without modifying the large fixture."""
        data = self._get_input(node)
        self.artifact_store.save(node.node_id, data)
        return node.node_id, {}

    monkeypatch.setattr(PipelineEngine, "_run_transformer", passthrough)
    nodes = [_source()]
    for index in range(20):
        nodes.append(_node(f"step_{index}", inputs=[nodes[-1]["node_id"]]))
    shared_leaf = nodes[-1]["node_id"]
    nodes.extend([_node("branch_a", inputs=[shared_leaf]), _node("branch_b", inputs=[shared_leaf])])
    response = _preview(preview_client, nodes, selected=None, inspect_all=True)
    entries = response["node_inspections"]
    assert len(entries) == 44
    tables = [
        table
        for entry in entries
        for side in ("input", "output")
        for table in entry[side]["tables"]
    ]
    assert len(tables) == 86
    assert all(table["row_count"] == 100 and table["column_count"] == 100 for table in tables)
    assert all(len(table["columns"]) == 100 and table["truncated"] for table in tables)
    assert all(table["rows"] for table in tables)
    assert len({len(table["rows"]) for table in tables}) == 1
    sample_bytes = sum(len(json.dumps(table["rows"], separators=(",", ":"))) for table in tables)
    assert sample_bytes <= 8 * 1024 * 1024
    assert all(
        len(json.dumps(table["rows"], separators=(",", ":"))) <= 256 * 1024 for table in tables
    )


def test_split_tables_share_reduced_json_budget():
    """Split table array delimiters must fit the assigned side allowance too."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    frame = pd.DataFrame({"x": ["a"]})
    side = snapshot_side(
        SplitDataset((frame, frame), (frame, frame), (frame, frame)), "output", sample_budget=64
    )
    tables = side["tables"]
    assert len(tables) == 6
    assert all(table["row_count"] == 1 and table["columns"] for table in tables)
    assert sum(len(json.dumps(table["rows"], separators=(",", ":"))) for table in tables) <= 64


def test_zero_sample_budget_retains_shape_and_schema():
    """Exhausted sample allowance must leave a usable measured schema with honest truncation."""
    from backend.ml_pipeline._execution.engine._inspection import snapshot_side

    side = snapshot_side(pd.DataFrame({"value": [1, 2]}), "output", sample_budget=0)
    table = side["tables"][0]
    assert side["status"] == "available"
    assert table["row_count"] == 2
    assert table["columns"] == [{"name": "value", "dtype": "int64"}]
    assert table["rows"] == []
    assert table["truncated"] is True


def test_shared_data_path_identity_ignores_downstream_training_branches(preview_client):
    """Two training consumers must not invent two preprocessing paths for their shared input."""
    response = _preview(
        preview_client,
        [
            _source(),
            _node("scale", inputs=["source"], params={"_display_name": "Scale values"}),
            _node("model_a", "training", ["scale"], {"_display_name": "Downstream model A"}),
            _node("model_b", "training", ["scale"], {"_display_name": "Downstream model B"}),
        ],
        selected=None,
        inspect_all=True,
    )
    entries = response["node_inspections"]
    assert len(entries) == 6
    for node_id in ("source", "scale"):
        copies = [entry for entry in entries if entry["node_id"] == node_id]
        assert len(copies) == 2 and copies[0]["branch_id"] != copies[1]["branch_id"]
        assert copies[0]["path_id"] and copies[0]["path_id"] == copies[1]["path_id"]
        assert "Downstream model" not in copies[0]["path_label"]
    assert (
        "Scale values"
        in next(entry for entry in entries if entry["node_id"] == "scale")["path_label"]
    )


def test_parallel_terminal_keeps_distinct_upstream_data_paths(preview_client):
    """Different upstream subgraphs must remain distinct even when their measured samples match."""
    response = _preview(
        preview_client,
        [
            _source(),
            _node("left", inputs=["source"], params={"_display_name": "Left scaling"}),
            _node("right", inputs=["source"], params={"_display_name": "Right scaling"}),
            _node("model", "training", ["left", "right"], {"execution_mode": "parallel"}),
        ],
        selected=None,
        inspect_all=True,
    )
    models = [entry for entry in response["node_inspections"] if entry["node_id"] == "model"]
    assert len(models) == 2
    assert models[0]["output"] == models[1]["output"]
    assert models[0]["path_id"] and models[0]["path_id"] != models[1]["path_id"]
    assert any("Left scaling" in entry["path_label"] for entry in models)
    assert any("Right scaling" in entry["path_label"] for entry in models)


def test_data_path_identity_ignores_descendants_pipeline_ids_and_display_names(preview_client):
    """A downstream fork, generated run identity or renamed label must not change upstream provenance."""
    original = _preview(
        preview_client, [_source(), _node("scale", inputs=["source"])], pipeline_id="first"
    )["node_inspections"][0]
    changed = _preview(
        preview_client,
        [
            _source(),
            _node("scale", inputs=["source"], params={"_display_name": "x" * 1000}),
            _node("tail_a", inputs=["scale"]),
            _node("tail_b", inputs=["scale"]),
        ],
        pipeline_id="second",
    )["node_inspections"]
    assert len(changed) == 2
    assert original["path_id"] and all(entry["path_id"] == original["path_id"] for entry in changed)
    assert all(entry["path_label"] and len(entry["path_label"]) <= 240 for entry in changed)


def test_data_path_identity_preserves_input_order(preview_client):
    """Fan-in edge order belongs to provenance because last-wins merge semantics depend on it."""
    first = _preview(
        preview_client, [_source("a"), _source("b"), _node("scale", inputs=["a", "b"])]
    )["node_inspections"][0]
    second = _preview(
        preview_client, [_source("a"), _source("b"), _node("scale", inputs=["b", "a"])]
    )["node_inspections"][0]
    assert first["input"] == second["input"]
    assert first["path_id"] and first["path_id"] != second["path_id"]


def test_data_path_identity_keeps_upstream_types_and_parameters(preview_client):
    """Equal sampled values cannot erase different preprocessing configurations from provenance."""
    preview_client.catalog.load.return_value = pd.DataFrame({"value": [0.0, 0.0]})
    receipts = []
    for kind, params in (
        ("StandardScaler", {"with_mean": True}),
        ("StandardScaler", {"with_mean": False}),
        ("MinMaxScaler", {}),
    ):
        response = _preview(
            preview_client,
            [
                _source(),
                _node("upstream", kind, ["source"], params),
                _node("scale", inputs=["upstream"]),
            ],
        )
        receipts.append(response["node_inspections"][0])
    assert all(receipt["output"]["status"] == "available" for receipt in receipts)
    assert all(receipt["output"] == receipts[0]["output"] for receipt in receipts)
    assert len({receipt["path_id"] for receipt in receipts}) == 3
