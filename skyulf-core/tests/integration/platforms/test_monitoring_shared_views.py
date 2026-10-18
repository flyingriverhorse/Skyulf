"""Shared dashboard rules preserve observation identity and safe store upgrades."""

import json
import sqlite3

import pytest
from test_monitoring_bootstrap import existing_store, mutations

from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

SHARED = {"complete_observations", "monitoring_identities", "serving_windows"}


def test_shared_views_keep_policy_only_identities_without_a_full_measurement():
    """Late labels must not hide identities or move their last full observation time."""
    views = store.monitoring_views("ops.monitoring")
    assert views.keys() >= SHARED
    with sqlite3.connect(":memory:") as connection:
        connection.create_function(
            "get_json_object",
            2,
            lambda value, path: (
                json.dumps(json.loads(value)[path[2:]])
                if value and json.loads(value).get(path[2:]) is not None
                else None
            ),
        )
        connection.create_function("concat", -1, lambda *parts: "".join(map(str, parts)))
        identity = (
            "monitor_id TEXT, model_catalog TEXT, model_schema TEXT, model_name TEXT, "
            "model_version TEXT, environment TEXT, project TEXT, measured_at INTEGER"
        )
        connection.execute(f"CREATE TABLE current_health ({identity})")
        connection.execute(
            f"CREATE TABLE monitoring_results ({identity}, status TEXT, report_json TEXT)"
        )
        connection.execute(
            "INSERT INTO current_health VALUES ('new','cat','schema','cat.schema.new',"
            "NULL,'prod','one',NULL)"
        )
        payloads = [
            ("mixed", "1", "prod", 10, "healthy", '{"current_rows": 0}'),
            ("mixed", "1", "prod", 20, "no_data", '{"performance": {}}'),
            ("only-policy", None, "test", 30, "no_data", '{"performance": {}}'),
            ("mixed", "1", "test", 40, "failed", "{}"),
            ("zero", "2", "prod", 50, "no_data", '{"current_rows":0,"performance":{}}'),
            ("unknown", "3", "prod", 60, None, '{"performance":{}}'),
        ]
        for monitor, version, environment, measured, status, payload in payloads:
            connection.execute(
                "INSERT INTO monitoring_results VALUES (?, 'cat', 'schema', "
                "'cat.schema.model', ?, ?, 'one', ?, ?, ?)",
                (monitor, version, environment, measured, status, payload),
            )
        for name in ("complete_observations", "monitoring_identities"):
            query = views[name].replace("`ops`.`monitoring`.", "")
            connection.execute(f"CREATE VIEW {name} AS {query}")
        assert connection.execute(
            "SELECT monitor_id, measured_at FROM complete_observations ORDER BY measured_at"
        ).fetchall() == [("mixed", 10), ("mixed", 40), ("zero", 50)]
        identities = {
            (row[0], row[1], row[2]): row[3]
            for row in connection.execute(
                "SELECT monitor_id, model_version, monitor_context, latest_measurement "
                "FROM monitoring_identities"
            )
        }
        assert identities == {
            ("new", None, "prod / one / new"): None,
            ("mixed", "1", "prod / one / mixed"): 10,
            ("mixed", "1", "test / one / mixed"): 40,
            ("only-policy", None, "test / one / only-policy"): None,
            ("zero", "2", "prod / one / zero"): 50,
            ("unknown", "3", "prod / one / unknown"): None,
        }


def test_additive_upgrade_creates_shared_views_without_replacing_existing_objects():
    """Existing producers must gain the new reads without replacing another owner's views."""
    spark = existing_store()
    for name in SHARED:
        spark.objects.pop(f"ops.monitoring.{name}", None)
    store.ensure_monitoring_store(spark, "ops", "monitoring")
    created = [sql for sql in mutations(spark) if sql.startswith("CREATE VIEW")]
    assert len(created) == 3
    assert all("IF NOT EXISTS" in sql for sql in created)
    assert not any("REPLACE" in sql or "ALTER" in sql for sql in mutations(spark))
    assert {name.rsplit(".", 1)[1] for name in spark.objects} >= SHARED


@pytest.mark.parametrize("operation", ["ensure_monitoring_store", "initialize_monitoring_store"])
@pytest.mark.parametrize("view", sorted(SHARED))
def test_foreign_shared_view_blocks_all_store_mutations(operation, view):
    """An upgrade must preflight every new name before creating or changing sibling objects."""
    spark = existing_store()
    spark.objects[f"ops.monitoring.{view}"] = {"owner": "foreign", "schema": "view"}
    with pytest.raises(ValueError, match="ownership"):
        getattr(store, operation)(spark, "ops", "monitoring")
    assert mutations(spark) == []


def test_shared_views_follow_dependency_order_and_have_no_dashboard_parameters():
    """Fresh setup must build all dependencies before any view is analyzed by Spark."""
    views = store.monitoring_views("ops.monitoring")
    names = list(views)
    assert names.index("complete_observations") < names.index("current_health")
    assert names.index("current_health") < names.index("monitoring_identities")
    assert names.index("complete_observations") < names.index("serving_windows")
    assert all(":serving_" not in views[name] for name in SHARED)
