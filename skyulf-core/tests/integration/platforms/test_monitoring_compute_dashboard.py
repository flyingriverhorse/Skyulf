"""Execute the shipped compute aggregation against bounded numeric fixtures."""

import json
import sqlite3
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"


def _compute_sql(dataset: str) -> str:
    """Retain production interval union and weighting while adapting Spark syntax."""
    dashboard = json.loads((TEMPLATE / "src/monitoring/monitoring.lvdash.json").read_text())
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    assert dataset in datasets, "Execution needs a query backed by real node telemetry"
    sql = datasets[dataset]
    # Fixtures enter after task selection/explode; actual union, node scope and
    # weighted aggregation remain executable. Live Databricks probes cover the
    # dialect-specific selection/explode and serverless status expressions.
    sql = (
        "WITH bounds AS (SELECT 0 AS window_start, 300 AS window_end), prior_intervals AS ("
        + sql.split("prior_intervals AS (", 1)[1]
    )
    sql = sql.replace("system.compute.node_timeline", "node_samples")
    sql = sql.replace("b.window_start - INTERVAL 1 MINUTE", "b.window_start - 60")
    sql = sql.replace("timestampdiff(MICROSECOND,", "timestampdiff('MICROSECOND',")
    if dataset.endswith("summary"):
        sql = sql.split(" CASE WHEN coalesce(:execution_workspace", 1)[0].rstrip()
        sql = sql.removesuffix(",") + " FROM weighted_samples"
    return sql


def _query(dataset: str, *, empty: bool = False) -> list[sqlite3.Row]:
    """Keep duplicate intervals, partial minutes and foreign identities in fixtures."""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.create_function("greatest", -1, max)
        connection.create_function("least", -1, min)
        connection.create_function("timestampdiff", 3, lambda unit, start, end: (end - start) * 1e6)
        connection.executescript(
            """
            CREATE TABLE task_intervals (
                account_id TEXT, workspace_id TEXT, cluster_id TEXT,
                interval_start INTEGER, interval_end INTEGER
            );
            CREATE TABLE node_samples (
                account_id TEXT, workspace_id TEXT, cluster_id TEXT, instance_id TEXT,
                start_time INTEGER, end_time INTEGER, driver BOOLEAN,
                cpu_user_percent REAL, cpu_system_percent REAL, mem_used_percent REAL
            );
            INSERT INTO task_intervals VALUES
                ('a', 'w', 'c', 0, 40), ('a', 'w', 'c', 0, 40),
                ('a', 'w', 'c', 10, 15), ('a', 'w', 'c', 20, 50),
                ('a', 'w', 'c', 60, 70), ('a', 'w', 'c', 80, 100);
            INSERT INTO node_samples VALUES
                ('a', 'w', 'c', 'n1', 0, 60, 1, 8, 2, 20),
                ('a', 'w', 'c', 'n1', 0, 60, 1, 8, 2, 20),
                ('a', 'w', 'c', 'n2', 0, 60, 0, 25, 5, 40),
                ('a', 'w', 'c', 'n1', 60, 120, 1, 85, 5, 80),
                ('a', 'w', 'c', 'n3', 0, 60, 0, NULL, NULL, NULL),
                ('a', 'other', 'c', 'n1', 0, 60, 1, 99, 0, 99),
                ('other', 'w', 'c', 'n1', 0, 60, 1, 99, 0, 99),
                ('a', 'w', 'other', 'n1', 0, 60, 1, 99, 0, 99);
            """
        )
        if empty:
            connection.execute("DELETE FROM node_samples")
        return connection.execute(
            _compute_sql(dataset), {"execution_workspace": "w", "execution_job": "j"}
        ).fetchall()


def test_compute_summary_deduplicates_overlap_and_weights_partial_minutes() -> None:
    """Concurrent tasks must not double count nodes or turn missing metrics into zeros."""
    rows = _query("execution_compute_summary")
    assert len(rows) == 1
    assert rows[0]["avg_cpu"] == pytest.approx(4700 / 130 / 100)
    assert rows[0]["avg_ram"] == pytest.approx(5400 / 130 / 100)
    assert rows[0]["peak_cpu"] == pytest.approx(0.9)
    assert rows[0]["peak_ram"] == pytest.approx(0.8)
    assert rows[0]["measured_clusters"] == 1


def test_compute_timeline_preserves_node_minute_peaks() -> None:
    """Per-minute node averages must retain unequal peaks and exclude foreign IDs."""
    rows = _query("execution_compute_timeline")
    assert len(rows) == 2
    assert [row["avg_cpu"] for row in rows] == pytest.approx([0.2, 0.9])
    assert [row["avg_ram"] for row in rows] == pytest.approx([0.3, 0.8])
    assert [row["peak_cpu"] for row in rows] == pytest.approx([0.3, 0.9])
    assert [row["peak_ram"] for row in rows] == pytest.approx([0.4, 0.8])


def test_compute_missing_telemetry_stays_null() -> None:
    """An empty classic-node source must never display a fabricated zero utilization."""
    rows = _query("execution_compute_summary", empty=True)
    assert len(rows) == 1
    assert all(rows[0][field] is None for field in ("avg_cpu", "peak_cpu", "avg_ram", "peak_ram"))
    assert rows[0]["measured_clusters"] == 0
    assert _query("execution_compute_timeline", empty=True) == []
