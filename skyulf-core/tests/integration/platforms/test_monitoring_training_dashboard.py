"""Training evidence tables retain shared scope and explicit missing evidence."""

import json
import re
import sqlite3
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
SECTIONS = {
    "training_candidates": "candidates",
    "training_parameters": "parameters",
    "training_datasets": "datasets",
}


def _dashboard() -> dict:
    """Read the exact dashboard payload installed by generated bundles."""
    return json.loads(
        (TEMPLATE / "src/monitoring/monitoring.lvdash.json").read_text(encoding="utf-8")
    )


def _dataset(name: str) -> dict:
    """Require a named production dataset without substituting fixture SQL."""
    datasets = {item["name"]: item for item in _dashboard()["datasets"]}
    assert name in datasets, f"Missing training evidence dataset: {name}"
    return datasets[name]


def _sqlite_sql(name: str) -> str:
    """Translate only Spark's typed outer explode and struct access for SQLite."""
    sql = "".join(_dataset(name)["queryLines"])
    sql, replacements = re.subn(
        r"LATERAL VIEW OUTER explode\(from_json\((get_json_object\([^\n]+?\)),\s*"
        r"'array<struct<[^']+>>'\)\) \w+ AS evidence",
        r"LEFT JOIN json_each(\1) AS evidence ON 1 = 1",
        sql,
    )
    assert replacements == 1, "Training rows must use typed, identity-preserving outer explode"
    return re.sub(r"evidence\.(\w+)", r"json_extract(evidence.value, '$.\1')", sql)


def _query(name: str, summaries: list[tuple], **scope: str) -> list[dict]:
    """Execute real dataset scope/status/link expressions over isolated identities."""
    sql = _sqlite_sql(name)
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "get_json_object",
            2,
            lambda value, path: connection.execute(
                "SELECT json_extract(?, ?)", (value, path)
            ).fetchone()[0],
        )
        connection.create_function("concat", -1, lambda *parts: "".join(map(str, parts)))
        connection.execute(
            "CREATE TABLE model_version_details (monitor_id TEXT, model_catalog TEXT, "
            "model_schema TEXT, model_name TEXT, model_version TEXT, monitor_context TEXT, "
            "report_id TEXT, training_summary_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO model_version_details VALUES (?, 'catalog', 'schema', ?, ?, ?, ?, ?)",
            summaries,
        )
        parameters = {
            "scope_model": "All models",
            "scope_version": "All versions",
            "scope_monitor": "All contexts",
            **scope,
        }
        return [dict(row) for row in connection.execute(sql, parameters)]


def _summary(section: str, rows: list[dict], status: str = "recorded") -> str:
    """Supply independently specified rows in the collector's public wire shape."""
    return json.dumps(
        {
            "format_version": 1,
            section: {"status": status, "reason": f"{status} reason", "rows": rows},
        }
    )


def test_training_evidence_reuses_global_scope_parameters() -> None:
    """Changing any shared identity control must also change all three training tables."""
    dashboard = _dashboard()
    shared = next(page for page in dashboard["pages"] if page["name"] == "investigation_filters")
    defaults = {
        "scope_model": "All models",
        "scope_version": "All versions",
        "scope_monitor": "All contexts",
    }
    for name in SECTIONS:
        parameters = _dataset(name)["parameters"]
        assert {
            p["keyword"]: p["defaultSelection"]["values"]["values"][0]["value"] for p in parameters
        } == defaults
        for item in shared["layout"]:
            control = item["widget"]
            query = next(q for q in control["queries"] if q["query"]["datasetName"] == name)
            binding = query["query"]["parameters"][0]
            assert binding["keyword"] in defaults
            assert {"parameterName": binding["name"], "queryName": query["name"]} in control[
                "spec"
            ]["encodings"]["fields"]


@pytest.mark.parametrize("name,section", SECTIONS.items())
def test_training_rows_keep_unobserved_and_missing_identities(name: str, section: str) -> None:
    """Empty training evidence must remain visible without borrowing another version's rows."""
    rows = _query(
        name,
        [
            ("observed", "model-a", "4", "batch", "r4", _summary(section, [{}])),
            ("old", "model-a", "3", "batch", "r3", None),
            ("waiting", "model-b", None, "serving", None, None),
            ("absent", "model-c", "1", "batch", "r1", _summary(section, [], "unavailable")),
        ],
    )
    by_monitor = {row["monitor_id"]: row for row in rows}
    assert len(rows) == 4
    assert {key: row["evidence_status"] for key, row in by_monitor.items()} == {
        "observed": "recorded",
        "old": "not_recorded",
        "waiting": "never_observed",
        "absent": "unavailable",
    }
    assert by_monitor["old"]["model_version"] == "3"
    assert by_monitor["waiting"]["model_version"] is None
    assert by_monitor["absent"]["evidence_reason"] == "unavailable reason"
    assert by_monitor["old"]["evidence_reason"]
    assert by_monitor["waiting"]["evidence_reason"]


@pytest.mark.parametrize("name,section", SECTIONS.items())
def test_training_scope_does_not_cross_models_versions_or_contexts(name: str, section: str) -> None:
    """A mismatched global selection must stay empty rather than use a nearby identity."""
    summaries = [
        ("match", "model-a", "4", "batch", "r1", _summary(section, [{}, {}])),
        ("version", "model-a", "3", "batch", "r2", _summary(section, [{}])),
        ("context", "model-a", "4", "serving", "r3", _summary(section, [{}])),
        ("model", "model-b", "4", "batch", "r4", _summary(section, [{}])),
    ]
    rows = _query(name, summaries, scope_model="model-a", scope_version="4", scope_monitor="batch")
    assert [row["monitor_id"] for row in rows] == ["match", "match"]
    assert _query(name, summaries, scope_model="model-b", scope_version="3") == []


def test_candidate_values_preserve_loser_identity_null_scores_and_bias_evidence() -> None:
    """A loser must not inherit registration or replace unknown scores with zero."""
    summary = _summary(
        "candidates",
        [
            {
                "candidate": "winner",
                "model_type": "RandomForestClassifier",
                "strategy": "cv",
                "registered_version": "4",
                "selected": True,
                "metric": "accuracy",
                "direction": "maximize",
                "score": 0.88,
                "score_std": 0.03,
                "evaluation_mode": "cross_validation",
                "unbiased_estimate": False,
                "selection_reason": "Highest saved score",
                "duration_seconds": 25.5,
                "run_id": "abc",
                "experiment_id": "42",
                "source": "training_report",
            },
            {"candidate": "loser", "selected": False, "registered_version": None, "score": None},
        ],
    )
    rows = _query("training_candidates", [("m", "model", "4", "batch", "r", summary)])
    by_candidate = {row["candidate"]: row for row in rows}
    assert by_candidate["loser"]["registered_version"] is None
    assert by_candidate["loser"]["model_version"] == "4"
    assert by_candidate["loser"]["score"] is None
    assert by_candidate["loser"]["duration_seconds"] is None
    winner = by_candidate["winner"]
    assert (winner["score"], winner["score_std"], winner["unbiased_estimate"]) == (0.88, 0.03, 0)
    assert winner["duration_seconds"] == 25.5
    assert winner["candidate_run_link"] == "/ml/experiments/42/runs/abc"


def test_parameter_rows_keep_json_values_without_coercion() -> None:
    """Chosen model parameters must retain booleans, nulls and structured JSON values."""
    summary = _summary(
        "parameters",
        [
            {
                "section": "model",
                "name": "class_weight",
                "value": '{"a":2,"b":1}',
                "source": "config",
            }
        ],
    )
    rows = _query("training_parameters", [("m", "model", "4", "batch", "r", summary)])
    assert rows[0]["value"] == '{"a":2,"b":1}'
    assert rows[0]["section"] == "model"
    assert rows[0]["source"] == "config"


def test_dataset_inputs_links_require_verified_inputs_and_complete_run_identity() -> None:
    """Stored lineage remains visible even when an Inputs run link cannot be verified."""
    rows: list[dict] = [
        {"role": "train", "inputs_status": "recorded", "run_id": "abc", "experiment_id": "42"},
        {"role": "test", "inputs_status": "unavailable", "run_id": "abc", "experiment_id": "42"},
        {"role": "reference", "inputs_status": "recorded", "run_id": "abc", "experiment_id": None},
    ]
    for row in rows:
        row.update({"source_table": "catalog.schema.source", "source_version": 3, "row_count": 0})
    result = _query(
        "training_datasets", [("m", "model", "4", "batch", "r", _summary("datasets", rows))]
    )
    by_role = {row["role"]: row for row in result}
    assert by_role["train"]["inputs_run_link"] == "/ml/experiments/42/runs/abc"
    assert by_role["test"]["inputs_run_link"] is None
    assert by_role["reference"]["inputs_run_link"] is None
    assert all(row["source_table"] == "catalog.schema.source" for row in result)
    assert all(row["source_version"] == 3 and row["row_count"] == 0 for row in result)


def test_training_tables_expose_evidence_with_native_table_links_and_hover_details() -> None:
    """Operators must see evidence status, CV limitations and lineage details in the page."""
    page = next(page for page in _dashboard()["pages"] if page["name"] == "training_lineage")
    widgets = {item["widget"]["name"]: item for item in page["layout"]}
    required = {
        "training_candidates": {
            "candidate",
            "model_type",
            "strategy",
            "registered_version",
            "selected",
            "metric",
            "direction",
            "score",
            "score_std",
            "evaluation_mode",
            "unbiased_estimate",
            "selection_reason",
            "duration_seconds",
            "run_id",
            "experiment_id",
            "source",
        },
        "training_parameters": {"section", "name", "value", "source"},
        "training_datasets": {
            "role",
            "name",
            "source_table",
            "source_version",
            "delta_table_id",
            "uc_table_id",
            "dataset_digest",
            "row_count",
            "membership_digest",
            "run_id",
            "experiment_id",
            "feature_binding",
            "source",
            "inputs_status",
        },
    }
    for name, fields in required.items():
        assert name in widgets
        item = widgets[name]
        widget = item["widget"]
        assert widget["spec"]["version"] == 2
        assert widget["spec"]["widgetType"] == "table"
        assert item["position"]["y"] < widgets["training_alias_capture"]["position"]["y"]
        columns = widget["spec"]["encodings"]["columns"]
        shown = {column["fieldName"] for column in columns}
        for column in columns:
            shown.update(
                re.findall(r"{{(\w+)}}", column.get("tooltip", {}).get("templatedText", ""))
            )
        assert (
            fields
            | {
                "model_name",
                "model_version",
                "monitor_context",
                "evidence_status",
                "evidence_reason",
            }
            <= shown
        )
        query = widget["queries"][0]["query"]
        assert query["datasetName"] == name and query["disaggregated"] is True
        assert shown <= {field["name"] for field in query["fields"]}
    candidate_columns = widgets["training_candidates"]["widget"]["spec"]["encodings"]["columns"]
    assert (
        next(c for c in candidate_columns if c["fieldName"] == "model_version")["displayName"]
        == "Monitored version"
    )
    assert (
        next(c for c in candidate_columns if c["fieldName"] == "registered_version")["displayName"]
        == "Candidate registered version"
    )
    assert next(c for c in candidate_columns if c["fieldName"] == "run_id")["link"] == {
        "templatedURL": "{{candidate_run_link}}"
    }
    dataset_columns = widgets["training_datasets"]["widget"]["spec"]["encodings"]["columns"]
    assert next(c for c in dataset_columns if c["fieldName"] == "run_id")["link"] == {
        "templatedURL": "{{inputs_run_link}}"
    }
