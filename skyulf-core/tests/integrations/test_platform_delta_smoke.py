"""Exercise the reusable Delta publication probe against real local transactions."""

import runpy
from pathlib import Path
from uuid import uuid4

import pytest


def test_delta_smoke_records_commits_and_retains_only_its_tables(delta_spark, monkeypatch):
    """A known regression bundle must publish, replay and explicitly clear January safely."""
    examples = Path(__file__).parents[2] / "examples"
    probe = examples / "databricks_delta_smoke.py"
    assert probe.is_file(), "The reusable Delta smoke example is missing."
    run_smoke = runpy.run_path(str(probe))["run_smoke"]
    identity = uuid4()
    monkeypatch.setitem(run_smoke.__globals__, "uuid4", lambda: identity)
    owned = [
        f"default.skyulf_delta_smoke_{identity.hex}_{role}"
        for role in ("source", "target", "control")
    ]
    build_bundle = runpy.run_path(str(examples / "databricks_batch_smoke.py"))["build_gold_bundle"]
    bundle, _ = build_bundle("pandas")
    try:
        report = run_smoke(
            delta_spark,
            namespace="default",
            bundle=bundle,
            model_name="smoke_regression",
            model_version="1",
        )
        assert report["stage"] == "delta_publication"
        assert report["platform_gate_complete"] is False
        assert report["checks"] == {
            "publication": True,
            "replay": True,
            "stale_rejected": True,
            "empty_rejected": True,
            "empty_replacement": True,
            "empty_replay": True,
            "ownership_released": True,
            "prior_period_preserved": True,
        }
        assert report["model_digest"] == bundle.semantic_digest
        assert report["model_name"] == "smoke_regression"
        assert report["model_version"] == "1"
        assert report["receipts"] == {
            "first": {"commit_version": 1, "input_count": 2, "output_count": 2, "replayed": False},
            "replay": {"commit_version": 1, "input_count": 2, "output_count": 2, "replayed": True},
            "empty": {"commit_version": 2, "input_count": 0, "output_count": 0, "replayed": False},
            "empty_replay": {
                "commit_version": 2,
                "input_count": 0,
                "output_count": 0,
                "replayed": True,
            },
        }
        tables = report["tables"]
        assert set(tables) == {"source", "target", "control"}
        assert tables["source"]["version"] == 1
        assert tables["target"]["version"] == 2
        for entry in tables.values():
            assert delta_spark.catalog.tableExists(entry["name"])
            assert entry["count"] == 1
            assert delta_spark.sql(f"DESCRIBE DETAIL {entry['name']}").first()["id"] == entry["id"]
        target_row = delta_spark.table(tables["target"]["name"]).first()
        assert (target_row.id, target_row.prediction) == (9, 99.0)
        control_row = delta_spark.table(tables["control"]["name"]).first()
        assert control_row.target_id == tables["target"]["id"]
        assert control_row.owner is None
    finally:
        for table in owned:
            delta_spark.sql(f"DROP TABLE IF EXISTS {table}").collect()


@pytest.mark.parametrize("namespace", ["a.b.c", "default; DROP TABLE x", "", "a..b"])
def test_delta_smoke_rejects_invalid_namespace_before_io(namespace):
    """Invalid catalog qualification must fail before any Spark table operation."""
    probe = Path(__file__).parents[2] / "examples" / "databricks_delta_smoke.py"
    assert probe.is_file(), "The reusable Delta smoke example is missing."
    run_smoke = runpy.run_path(str(probe))["run_smoke"]
    with pytest.raises(ValueError, match="namespace"):
        run_smoke(
            None, namespace=namespace, bundle=None, model_name="regression", model_version="1"
        )
